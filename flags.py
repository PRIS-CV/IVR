import argparse

DATA_FOLDER = "ROOT_FOLDER"

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--config', default='configs/args.yml', help='path of the config file')
parser.add_argument('--dataset', default='zappos', help='utzappos|clothing|aoclevr')
parser.add_argument('--data_dir', default='ut-zap50k', help='local path to data root dir from ' + DATA_FOLDER)
parser.add_argument('--logpath', default=None, help='Path to dir where to logs are stored (test only)')
parser.add_argument('--splitname', default='compositional-split-natural', help="dataset split")
parser.add_argument('--cv_dir', default='logs/', help='dir to save checkpoints and logs to')
parser.add_argument('--name', default='temp', help='Name of exp used to name models')
parser.add_argument('--load', default=None, help='path to checkpoint to load from')
parser.add_argument('--image_extractor', default = 'resnet18', help = 'Feature extractor model')
parser.add_argument('--test_set', default='val', help='val|test mode')
parser.add_argument('--test_batch_size', type=int, default=32, help="Batch size at test/eval time")
parser.add_argument('--cpu_eval', action='store_true', help='Perform test on cpu')
parser.add_argument('--seed', type=int, default=0, help='seed')

# Model parameters
parser.add_argument('--emb_dim', type=int, default=300, help='dimension of share embedding space')
parser.add_argument('--nlayers', type=int, default=3, help='Layers in the image embedder')
parser.add_argument('--bias', type=float, default=1e3, help='Bias value for unseen concepts')
parser.add_argument('--update_features', action = 'store_true', default=False, help='If specified, train feature extractor')
parser.add_argument('--cosine_scale', type=float, default=20,help="Scale for cosine similarity")
parser.add_argument('--drop', type=float,default=5/6, help='drop rate')
parser.add_argument('--lambda_rep', type=float, default=1.0, help='weight of rep losses at the representation level')
parser.add_argument('--lambda_grad', type=float, default=10.0, help='weight of grad losses at the gradient level')

# Hyperparameters
parser.add_argument('--topk', type=int, default=1,help="Compute topk accuracy")
parser.add_argument('--workers', type=int, default=8,help="Number of workers")
parser.add_argument('--batch_size', type=int, default=512,help="Training batch size")
parser.add_argument('--lr', type=float, default=5e-5,help="Learning rate")
parser.add_argument('--lrg', type=float, default=1e-3,help="Learning rate feature extractor")
parser.add_argument('--wd', type=float, default=5e-5,help="Weight decay")
parser.add_argument('--save_every', type=int, default=10000,help="Frequency of snapshots in epochs")
parser.add_argument('--eval_val_every', type=int, default=1,help="Frequency of eval in epochs")
parser.add_argument('--max_epochs', type=int, default=800,help="Max number of epochs")
