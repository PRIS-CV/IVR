import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.common import MLP
import numpy as np
import torch.autograd as autograd

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Disentangler(nn.Module):
    def __init__(self, args):
        super(Disentangler, self).__init__()
        self.fc1 = nn.Linear(args.emb_dim, args.emb_dim)
        self.bn1_fc = nn.BatchNorm1d(args.emb_dim)

    def forward(self, x):
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        return x


class IVR(nn.Module):

    def __init__(self, dset, args): 
        super(IVR, self).__init__()
        self.args = args
        self.dset = dset
        def get_all_ids(relevant_pairs):
            attrs, objs = zip(*relevant_pairs)
            attrs = [dset.attr2idx[attr] for attr in attrs]
            objs = [dset.obj2idx[obj] for obj in objs]
            pairs = [a for a in range(len(relevant_pairs))]
            attrs = torch.LongTensor(attrs).to(device)
            objs = torch.LongTensor(objs).to(device)
            pairs = torch.LongTensor(pairs).to(device)
            return attrs, objs, pairs

        self.train_attrs, self.train_objs, self.train_pairs = get_all_ids(self.dset.train_pairs) 
        self.val_attrs, self.val_objs, self.val_pairs = get_all_ids(self.dset.pairs)

        self.scale = self.args.cosine_scale
        self.train_forward = self.train_forward_closed

        self.emb_dim=args.emb_dim
        self.image_embedder = MLP(dset.feat_dim, int(args.emb_dim), relu=False, num_layers=args.nlayers,
                                  dropout=True, norm=True, layers=[1024])

        self.attr_embedder = nn.Embedding(len(dset.attrs), 300)
        self.obj_embedder = nn.Embedding(len(dset.objs), 300)

        self.projection = nn.Linear(args.emb_dim * 2, args.emb_dim)

        self.D = nn.ModuleDict({'da': Disentangler(args), 'do': Disentangler(args)})
        self.attr_clf = MLP(args.emb_dim, len(dset.attrs), 1, relu = False)
        self.attr_clf_au = MLP(args.emb_dim, len(dset.attrs), 1, relu=False)
        self.obj_clf = MLP(args.emb_dim, len(dset.objs), 1, relu = False)
        self.obj_clf_au = MLP(args.emb_dim, len(dset.objs), 1, relu=False)

        self.drop = args.drop
        self.lambda_rep=args.lambda_rep
        self.lambda_grad=args.lambda_grad

    def compose(self, attrs, objs):
        attrs, objs = self.attr_embedder(attrs), self.obj_embedder(objs)
        inputs = torch.cat([attrs, objs], 1)
        output = self.projection(inputs)
        return output

    def train_forward_closed(self, x):
        img, attrs, objs, pairs = x[0], x[1], x[2], x[3]
        pos_attr_img, neg_objs, pos_obj_img, neg_attrs = x[4], x[5], x[6], x[7]
        neg_obj_pairs, neg_attr_pairs = x[10],x[11]

        img = self.image_embedder(img)
        pos_attr_img = self.image_embedder(pos_attr_img)
        pos_obj_img = self.image_embedder(pos_obj_img)

        pair_embed = self.compose(self.train_attrs, self.train_objs).permute(1, 0)

        img_normed = F.normalize(img, dim=1)
        pos_attr_img_normed = F.normalize(pos_attr_img, dim=1)
        pos_obj_img_normed = F.normalize(pos_obj_img, dim=1)
        pair_embed_normed = F.normalize(pair_embed, dim=1)

        pair_pred = torch.matmul(img_normed, pair_embed_normed)
        loss_comp = F.cross_entropy(self.scale * pair_pred, pairs)

        pair_pred_pa = torch.matmul(pos_attr_img_normed, pair_embed_normed)
        loss_comp_pa = F.cross_entropy(self.scale * pair_pred_pa, neg_obj_pairs)

        pair_pred_po = torch.matmul(pos_obj_img_normed,  pair_embed_normed)
        loss_comp_po = F.cross_entropy(self.scale * pair_pred_po, neg_attr_pairs)

        img_da = self.D['da'](img) 
        img_do = self.D['do'](img) 

        pos_attr_img_da = self.D['da'](pos_attr_img)
        pos_obj_img_do = self.D['do'](pos_obj_img)

        neg_attr_img_da = self.D['da'](pos_obj_img)
        neg_obj_img_do = self.D['do'](pos_attr_img)

        loss_neg_attr = F.cross_entropy(self.attr_clf(neg_attr_img_da), neg_attrs)
        loss_neg_obj = F.cross_entropy(self.obj_clf(neg_obj_img_do), neg_objs)

        img_da_p = self.attr_clf(img_da)
        img_da_pp = self.attr_clf_au(pos_attr_img_da)
        img_do_p = self.obj_clf(img_do)
        img_do_pp = self.obj_clf_au(pos_obj_img_do)

        loss_attr = F.cross_entropy(img_da_p, attrs)
        loss_pos_attr = F.cross_entropy(img_da_pp, attrs)
        loss_obj = F.cross_entropy(img_do_p, objs)
        loss_pos_obj = F.cross_entropy(img_do_pp, objs)
        loss = loss_attr + loss_obj + loss_pos_attr + loss_pos_obj + \
               loss_neg_attr + loss_neg_obj + loss_comp + loss_comp_pa + loss_comp_po

        attr_one_hot = torch.nn.functional.one_hot(attrs, len(self.dset.attrs))
        obj_one_hot = torch.nn.functional.one_hot(objs, len(self.dset.objs))

        img_da_g = autograd.grad((img_da_p * attr_one_hot).sum(), img_da, retain_graph=True)[0]
        img_da_g_p = autograd.grad((img_da_pp * attr_one_hot).sum(), pos_attr_img_da, retain_graph=True)[0]

        img_do_g = autograd.grad((img_do_p * obj_one_hot).sum(), img_do, retain_graph=True)[0]
        img_do_g_p = autograd.grad((img_do_pp * obj_one_hot).sum(), pos_obj_img_do, retain_graph=True)[0]

        diff_attr = torch.abs(img_da_g - img_da_g_p)
        perct_attr = torch.sort(diff_attr)[0][:, int(self.drop * self.emb_dim)]
        perct_attr = perct_attr.unsqueeze(1).repeat(1, self.emb_dim)
        mask_attr = diff_attr.lt(perct_attr.to(device)).float()

        diff_obj = torch.abs(img_do_g - img_do_g_p)
        perct_obj = torch.sort(diff_obj)[0][:, int(self.drop * self.emb_dim)]
        perct_obj = perct_obj.unsqueeze(1).repeat(1, self.emb_dim)
        mask_obj = diff_obj.lt(perct_obj.to(device)).float()

        attr_supp = self.attr_clf(img_da * mask_attr)
        obj_supp = self.obj_clf(img_do * mask_obj)

        attr_supp2 = self.attr_clf_au(pos_attr_img_da * mask_attr)
        obj_supp2 = self.obj_clf_au(pos_obj_img_do * mask_obj)

        loss_rep_attr1= F.cross_entropy(attr_supp, attrs)
        loss_rep_obj1 = F.cross_entropy(obj_supp, objs)

        loss_rep_attr2 = F.cross_entropy(attr_supp2, attrs)
        loss_rep_obj2 = F.cross_entropy(obj_supp2, objs)

        loss += self.lambda_rep*(loss_rep_attr1 + loss_rep_obj1+ loss_rep_attr2 + loss_rep_obj2)

        attr_grads=[]
        attr_env_loss=[loss_rep_attr1, loss_rep_attr2] 
        attr_clf=[self.attr_clf, self.attr_clf_au]
        for i in range(2):
            attr_env_grad = autograd.grad(attr_env_loss[i], attr_clf[i].parameters(), create_graph=True)
            attr_grads.append(attr_env_grad)
        loss_grad_attr = 1/2*(attr_grads[0][0] - attr_grads[1][0]).pow(2).sum() + 1/2*(attr_grads[0][1] - attr_grads[1][1]).pow(2).sum()

        obj_grads = []
        obj_env_loss=[loss_rep_obj1, loss_rep_obj2]
        obj_clf = [self.obj_clf, self.obj_clf_au]
        for i in range(2):
            obj_env_grad = autograd.grad(obj_env_loss[i], obj_clf[i].parameters(), create_graph=True) 
            obj_grads.append(obj_env_grad)
        loss_grad_obj = 1/2*(obj_grads[0][0] - obj_grads[1][0]).pow(2).sum() + 1/2*(obj_grads[0][1] - obj_grads[1][1]).pow(2).sum()

        loss += self.lambda_grad*(loss_grad_attr + loss_grad_obj)

        return loss, None


    def val_forward(self, x):
        img = x[0]
        img = self.image_embedder(img)
        
        pair_embeds = self.compose(self.val_attrs, self.val_objs).permute(1, 0)
        img_normed = F.normalize(img, dim=1)
        pair_embeds_normed = F.normalize(pair_embeds, dim=1)
        pair_pred = torch.matmul(img_normed, pair_embeds_normed)
  
        img_da = self.D['da'](img)
        img_do = self.D['do'](img)
        attr_pred = F.softmax(self.attr_clf(img_da), dim =1)
        obj_pred = F.softmax(self.obj_clf(img_do), dim = 1)
  
        scores = {}
        for itr, (attr, obj) in enumerate(self.dset.pairs):
            attr_id, obj_id = self.dset.attr2idx[attr], self.dset.obj2idx[obj]
            scores[(attr, obj)] = attr_pred[:, attr_id] * obj_pred[:, obj_id]+pair_pred[:, self.dset.all_pair2idx[(attr, obj)]]    
        return None, scores

    def forward(self, x):
        if self.training:
            loss, pred = self.train_forward(x)
        else:
            with torch.no_grad():
                loss, pred = self.val_forward(x)
        return loss, pred




