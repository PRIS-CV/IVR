import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.invariant_learning import IVR

from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import tqdm
from tqdm import tqdm
import os
from os.path import join as ospj
import csv

from data import dataset as dset
from utils.common import Evaluator
from utils.utils import save_args, load_args
from flags import parser, DATA_FOLDER

torch.multiprocessing.set_sharing_strategy('file_system')

best_auc = 0
best_hm = 0
best_unseen=0
compose_switch = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    args = parser.parse_args()
    load_args(args.config, args)
    logpath = os.path.join(args.cv_dir, args.name)
    os.makedirs(logpath, exist_ok=True)
    save_args(args, logpath, args.config)
    writer = SummaryWriter(log_dir = logpath, flush_secs = 30)
    
    seed=args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    trainset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER, args.data_dir),
        phase='train',
        split=args.splitname,
        model=args.image_extractor,
        update_features=args.update_features
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers)

    testset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER, args.data_dir),
        phase=args.test_set, 
        split=args.splitname,
        model=args.image_extractor,
        update_features=args.update_features
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers)


    if not args.update_features:
        image_extractor = None

    model = IVR(trainset, args)
    model = model.to(device)

    model_params = [param for name, param in model.named_parameters() if param.requires_grad]
    optim_params = [{'params':model_params}]
    optimizer = optim.Adam(optim_params, lr=args.lr, weight_decay=args.wd)

    train = train_normal
    evaluator_val = Evaluator(testset, model)

    start_epoch = 0
    if args.load is not None:
        checkpoint = torch.load(args.load)
        if image_extractor:
            try:
                image_extractor.load_state_dict(checkpoint['image_extractor'])
                if args.freeze_features:
                    print('Freezing image extractor')
                    image_extractor.eval()
                    for param in image_extractor.parameters():
                        param.requires_grad = False
            except:
                print('No Image extractor in checkpoint')
        model.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        print('Loaded model from ', args.load)

    for epoch in tqdm(range(start_epoch, args.max_epochs + 1), desc = 'Current epoch'):
        train(epoch, image_extractor, model, trainloader, optimizer, writer)

        if epoch % args.eval_val_every == 0:
            with torch.no_grad():
                test(epoch, image_extractor, model, testloader, evaluator_val, writer, args, logpath)
    print('Best AUC achieved is ', best_auc)
    print('Best HM achieved is ', best_hm)


def train_normal(epoch, image_extractor, model, trainloader, optimizer, writer):
    if image_extractor:
        image_extractor.train()
    model.train()

    train_loss = 0.0

    for idx, data in tqdm(enumerate(trainloader), total=len(trainloader), desc = 'Training'):
        data = [d.to(device) for d in data]

        if image_extractor:
            data[0] = image_extractor(data[0])

        loss, _ = model(data)
 
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        train_loss += loss.item()

    train_loss = train_loss/len(trainloader)
    writer.add_scalar('Loss/train_total', train_loss, epoch)

    print('Epoch: {}| Loss: {}'.format(epoch, round(train_loss, 2)))


def test(epoch, image_extractor, model, testloader, evaluator, writer, args, logpath):
    global best_auc, best_hm, best_unseen

    def save_checkpoint(filename):
        state = {
            'net': model.state_dict(),
            'epoch': epoch,
            'AUC': stats['AUC']
        }
        if image_extractor:
            state['image_extractor'] = image_extractor.state_dict()
        torch.save(state, os.path.join(logpath, 'ckpt_{}.t7'.format(filename)))

    if image_extractor:
        image_extractor.eval()

    model.eval()

    all_attr_gt, all_obj_gt, all_pair_gt, all_pred = [], [], [], []

    for idx, data in tqdm(enumerate(testloader), total=len(testloader), desc='Testing'):
        data = [d.to(device) for d in data]

        if image_extractor:
            data[0] = image_extractor(data[0])

        _, predictions= model(data)

        attr_truth, obj_truth, pair_truth =data[1], data[2], data[3]
        
        all_pred.append(predictions)
        all_attr_gt.append(attr_truth)
        all_obj_gt.append(obj_truth)
        all_pair_gt.append(pair_truth)

    if args.cpu_eval:
        all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt), torch.cat(all_obj_gt), torch.cat(all_pair_gt)
    else:
        all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt).to('cpu'), torch.cat(all_obj_gt).to(
            'cpu'), torch.cat(all_pair_gt).to('cpu')

    all_pred_dict = {}
    if args.cpu_eval:
        for k in all_pred[0].keys():
            all_pred_dict[k] = torch.cat(
                [all_pred[i][k].to('cpu') for i in range(len(all_pred))])
    else:
        for k in all_pred[0].keys():
            all_pred_dict[k] = torch.cat(
                [all_pred[i][k] for i in range(len(all_pred))])


    results = evaluator.score_model(all_pred_dict, all_obj_gt, bias=args.bias, topk=args.topk)
    stats = evaluator.evaluate_predictions(results, all_attr_gt, all_obj_gt, all_pair_gt, all_pred_dict, topk=args.topk)

    stats['a_epoch'] = epoch

    result = ''

    for key in stats:
        writer.add_scalar(key, stats[key], epoch)
        result = result + key + '  ' + str(round(stats[key], 4)) + '| '

    print(f'Test Epoch: {epoch}')
    print(result)

    if epoch > 0 and epoch % args.save_every == 0:
        save_checkpoint(epoch)
    if stats['AUC'] > best_auc:
        best_auc = stats['AUC']
        print('New best AUC ', best_auc)
        save_checkpoint('best_auc')

    if stats['best_hm'] > best_hm:
        best_hm = stats['best_hm']
        print('New best HM ', best_hm)
        save_checkpoint('best_hm')

    if stats['best_unseen'] > best_unseen:
        best_unseen = stats['best_unseen']
        print('New best_unseen ', best_unseen)
        save_checkpoint('best_unseen')

    with open(ospj(logpath, 'logs.csv'), 'a') as f:
        w = csv.DictWriter(f, stats.keys())
        if epoch == 0:
            w.writeheader()
        w.writerow(stats)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Best AUC achieved is ', best_auc)
        print('Best HM achieved is ', best_hm)