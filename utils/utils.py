import os
from os.path import join as ospj
import shutil
import sys
import yaml


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def save_args(args, log_path, argfile):
    shutil.copy('train.py', log_path)
    modelfiles = ospj(log_path, 'models')
    try:
        shutil.copy(argfile, log_path)
    except:
        print('Config exists')
    try:
        shutil.copytree('models/', modelfiles)
    except:
        print('Already exists')
    with open(ospj(log_path,'args_all.yaml'),'w') as f:
        yaml.dump(args, f, default_flow_style=False, allow_unicode=True)
    with open(ospj(log_path, 'args.txt'), 'w') as f:
        f.write('\n'.join(sys.argv[1:]))

def load_args(filename, args):
    with open(filename, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    for key, group in data_loaded.items():
        for key, val in group.items():
            setattr(args, key, val)