import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import third_party.reloc3r.croco.utils.misc as misc
# from third_party.reloc3r.croco.utils import misc as misc

from src.reloc3r_variants.data.utils import build_dataset
from src.reloc3r_variants.models.Reloc3rWithDiffusionHead import Reloc3rWithDiffusionHead


def get_args_parser():
    parser = argparse.ArgumentParser("Train Reloc3rWithDiffusionHead Model")

    # wandb arguments
    parser.add_argument('--entity', default="mtang4ucmerced-ucmerced")
    parser.add_argument('--project', default="Reloc3r_Variants")
    parser.add_argument('--exp_name', default="Reloc3rWithDiffusionHead")

    # model
    parser.add_argument('--model', default="Reloc3rWithDiffusionHead(img_size=512)",
                        help="string containing the model to build")
    parser.add_argument('--pretrained', default=None,
                        help="path of a starting checkpoint")
    
    # datasets
    parser.add_argument('--train_dataset', default="50000 @ MegaDepth(split='train', resolution=[(512, 384)])", type=str, help="training set")
    parser.add_argument('--test_dataset', default="ScanNet1500(resolution=(512, 384), seed=777) + MegaDepth_valid(split='test', resolution=(512, 384), seed=777)", type=str, help="testing set")
    
    # training
    parser.add_argument('--seed', default=42, type=int, help="Random seed")
    parser.add_argument('--batch_size', default=64, type=int,
                        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus)")
    parser.add_argument('--accum_iter', default=1, type=int,
                        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)")
    parser.add_argument('--epochs', default=800, type=int,
                        help="Maximum number of epochs for the scheduler")
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help="weight decay (default: 0.05)")
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR')

    parser.add_argument('--amp', type=int, default=0,
                        choices=[0, 1], help="Use Automatic Mixed Precision for pretraining")
    parser.add_argument("--disable_cudnn_benchmark", action='store_true', default=False,
                        help="set cudnn.benchmark = False")
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')

    # others
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # output dir
    parser.add_argument('--output_dir', default='./output/', type=str, help="path where to save the output")


    return parser


def train(args):
    ##### ENV Setup #####
    misc.init_distributed_mode(args)
    global_rank = misc.get_rank()
    world_size = misc.get_world_size()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    cudnn.benchmark = not args.disable_cudnn_benchmark

    seed = args.seed + global_rank
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("output_dir: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)


    ##### Prepare DataLoader #####
    print('Building train dataset {:s}'.format(args.train_dataset))
    data_loader_train = build_dataset(args.train_dataset, args.batch_size, args.num_workers, test=False)
    data_loader_test = {
        dataset.split('(')[0]: build_dataset(dataset, args.batch_size, args.num_workers, test=True)
        for dataset in args.test_dataset.split('+')
    }

    ##### Load Model #####
    # resume if last checkpoint exists
    last_ckpt_fname = os.path.join(args.output_dir, f'checkpoint-last.pth')
    args.resume = last_ckpt_fname if os.path.isfile(last_ckpt_fname) else None

    # build model
    print('Loading model: {:s}'.format(args.model))
    model = eval(args.model)
    model.to(device)
    model_without_ddp = model

    if args.pretrained and not args.resume:
        print('Loading pretrained: ', args.pretrained)
        ckpt = torch.load(args.pretrained, map_location=device)
        # modify ckpt keys for Reloc3rWithDiffusionHead
        pass





if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    train(args)