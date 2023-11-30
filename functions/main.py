
import numpy as np
import importlib
import os
import sys
import socket
conf_path = os.getcwd()
sys.path.append(conf_path)

from models import get_all_models
from argparse import ArgumentParser
from functions.args import add_management_args
from datasets import get_dataset
from models import get_model
from functions.training import train
from functions.conf import set_random_seed
from functions.create_if_not_exists import create_if_not_exists
from functions.distributed import make_ddp, make_dp
import torch

import uuid
import datetime

def lecun_fix():
    # Yann moved his website to CloudFlare. You need this now
    from six.moves import urllib
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

def parse_args():
    parser = ArgumentParser(description='mammoth', allow_abbrev=False)
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())

    torch.set_num_threads(4)
    add_management_args(parser)
    args = parser.parse_known_args()[0]
    mod = importlib.import_module('models.' + args.model)

    get_parser = getattr(mod, 'get_parser')
    parser = get_parser()
    args = parser.parse_args()

    if args.seed is not None:
        set_random_seed(args.seed)

    if args.model == 'mer': setattr(args, 'batch_size', 1)
    return args


def main(args=None):
    if torch.cuda.is_available():
        print("cuda is available")
    else:
        print("cuda is not available")
        exit(0)
    lecun_fix()
    if args is None:
        args = parse_args()

    os.putenv("MKL_SERVICE_FORCE_INTEL", "1")
    os.putenv("NPY_MKL_FORCE_INTEL", "1")

    # job number 
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()
    print("Start getting dataset!")
    dataset = get_dataset(args)
    print("Get dataset!")
    backbone = dataset.get_backbone()
    print("Get backbone")
    loss = dataset.get_loss()
    if args.model == 'joint':
        args.ignore_other_metrics=1
    model = get_model(args, backbone, loss, dataset.get_transform())
    if args.distributed == 'ddp':
        model.net = make_ddp(model.net)
        args.conf_ngpus = int(os.environ['MAMMOTH_WORLD_SIZE'])
    elif args.distributed == 'dp':
        model.net = make_dp(model.net)
        model.to('cuda:0')
        args.conf_ngpus = torch.cuda.device_count()
    elif args.distributed == 'no':
        args.distributed = None
    
    print("Training!")
    train(model, dataset, args)

if __name__ == '__main__':
    main()
