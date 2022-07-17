#external libs
import numpy as np
from tqdm import tqdm
from PIL import Image
import os
import random
from os.path import join as ospj
from glob import glob 
#torch libs
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
#local libs
from utils.utils import chunks
from models.image_extractor import get_image_extractor
from itertools import product
from random import choice

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ImageLoader:
    def __init__(self, root):
        self.root_dir = root

    def __call__(self, img):
        img = Image.open(ospj(self.root_dir,img)).convert('RGB')
        return img


def dataset_transform(phase):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    if phase == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    elif phase == 'val' or phase == 'test':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    return transform


class CompositionDataset(Dataset):
    def __init__(
        self,
        root,
        phase,
        split = 'compositional-split',
        model = 'resnet18',
        update_features = False,
    ):
        self.root = root
        self.phase = phase
        self.split = split
        self.update_features = update_features
        self.feat_dim = 512 if 'resnet18' in model else 2048

        self.attrs, self.objs, self.pairs, self.train_pairs, self.val_pairs, self.test_pairs = self.parse_split()
        self.train_data, self.val_data, self.test_data = self.get_split_info()
 
        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr : idx for idx, attr in enumerate(self.attrs)}

        if self.phase == 'train':
            self.pair2idx = {pair : idx for idx, pair in enumerate(self.train_pairs)}
        else:
            self.pair2idx = {pair : idx for idx, pair in enumerate(self.pairs)}
        self.all_pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}
        
        if self.phase == 'train':
            self.data = self.train_data
        elif self.phase == 'val':
            self.data = self.val_data
        elif self.phase == 'test':
            self.data = self.test_data
        
        self.all_data = self.train_data + self.val_data + self.test_data

        print('Dataset loaded')
        print('Train pairs: {}, Validation pairs: {}, Test Pairs: {}'.format(
            len(self.train_pairs), len(self.val_pairs), len(self.test_pairs)))
        print('Train images: {}, Validation images: {}, Test images: {}'.format(
            len(self.train_data), len(self.val_data), len(self.test_data)))

        self.sample_indices = list(range(len(self.data)))

        self.transform = dataset_transform(self.phase)
        self.loader = ImageLoader(ospj(self.root, 'images'))
        if not self.update_features:
            feat_file = ospj(root, model+'_features.t7')
            print(f'Using {model} and feature file {feat_file}')
            if not os.path.exists(feat_file):
                with torch.no_grad():
                    self.generate_features(feat_file, model)
            self.phase = phase
            activation_data = torch.load(feat_file)
            self.activations = dict(zip(activation_data['files'], activation_data['features']))
            self.feat_dim = activation_data['features'].size(1)
            print('{} activations loaded'.format(len(self.activations)))

    def parse_split(self):
        def parse_pairs(pair_list):
            with open(pair_list, 'r') as f:
                pairs = f.read().strip().split('\n')
                pairs = [line.split() for line in pairs]
                pairs = list(map(tuple, pairs))

            attrs, objs = zip(*pairs)
            return attrs, objs, pairs

        tr_attrs, tr_objs, tr_pairs = parse_pairs(ospj(self.root, self.split, 'train_pairs.txt'))
        vl_attrs, vl_objs, vl_pairs = parse_pairs(ospj(self.root, self.split, 'val_pairs.txt'))
        ts_attrs, ts_objs, ts_pairs = parse_pairs(ospj(self.root, self.split, 'test_pairs.txt'))
        
        all_attrs, all_objs = sorted(
            list(set(tr_attrs + vl_attrs + ts_attrs))), sorted(
                list(set(tr_objs + vl_objs + ts_objs)))
        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))
        return all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs

    def get_split_info(self):
        data = torch.load(ospj(self.root, 'metadata_{}.t7'.format(self.split)))
        train_data, val_data, test_data = [], [], []

        for instance in data:
            image, attr, obj, settype = instance['image'], instance['attr'], \
                instance['obj'], instance['set']
            curr_data = [image, attr, obj]
            if attr == 'NA' or (attr, obj) not in self.pairs or settype == 'NA':
                continue
            if settype == 'train':
                train_data.append(curr_data)
            elif settype == 'val':
                val_data.append(curr_data)
            else:
                test_data.append(curr_data)

        return train_data, val_data, test_data

    def generate_features(self, out_file, model):
        data = ospj(self.root,'images')
        files_before = glob(ospj(data, '**', '*.jpg'), recursive=True)
        files_all = []
        for current in files_before:
            parts = current.split('/')
            if "cgqa" in self.root:
                files_all.append(parts[-1])
            else:
                files_all.append(os.path.join(parts[-2],parts[-1]))
        transform = dataset_transform('test')
        feat_extractor = get_image_extractor(arch = model).eval()
        feat_extractor = feat_extractor.to(device)

        image_feats = []
        image_files = []
        for chunk in tqdm(
                chunks(files_all, 512), total=len(files_all) // 512, desc=f'Extracting features {model}'):

            files = chunk
            imgs = list(map(self.loader, files))
            imgs = list(map(transform, imgs))
            feats = feat_extractor(torch.stack(imgs, 0).to(device))
            image_feats.append(feats.data.cpu())
            image_files += files
        image_feats = torch.cat(image_feats, 0)
        print('features for %d images generated' % (len(image_files)))

        torch.save({'features': image_feats, 'files': image_files}, out_file)

    def __getitem__(self, index):
        index = self.sample_indices[index]
        image, attr, obj = self.data[index]
        if self.phase == 'train':
            positive_attr = self.same_A_diff_B(label_A=attr, label_B=obj, phase='attr')
            same_attr_image = positive_attr[0]
            one_obj=positive_attr[2]
            one_attr = positive_attr[1]
            positive_obj = self.same_A_diff_B(label_A=obj, label_B=attr, phase='obj')
            same_obj_image = positive_obj[0]
            two_attr=positive_obj[1]
            two_obj= positive_obj[2]

        if not self.update_features:
            img = self.activations[image]
            if self.phase == 'train':
                same_attr_img = self.activations[same_attr_image]
                same_obj_img = self.activations[same_obj_image]
        else:
            img = self.loader(image)
            img = self.transform(img)
            if self.phase == 'train':
                same_attr_img = self.loader(same_attr_image)
                same_attr_img = self.transform(same_attr_img)
                same_obj_img = self.loader(same_obj_image)
                same_obj_img = self.transform(same_obj_img)

        data = [img, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)]]

        if self.phase == 'train':
            data += [same_attr_img, self.obj2idx[one_obj], same_obj_img, self.attr2idx[two_attr],
                self.attr2idx[one_attr], self.obj2idx[two_obj],
                self.pair2idx[(attr, one_obj)], self.pair2idx[(two_attr, obj)]]

        return data


    def same_A_diff_B(self, label_A, label_B, phase='attr'):
        data1 = []
        for i in range(len(self.train_data)):
            if phase=='attr':
                if (self.train_data[i][1]== label_A) & (self.train_data[i][2] != label_B):
                    data1.append(self.train_data[i])
            else:
                if (self.train_data[i][2]== label_A) & (self.train_data[i][1] != label_B):
                    data1.append(self.train_data[i])
            
        data2 = choice(data1)
        return data2

    def __len__(self):
        return len(self.sample_indices)


