#!/usr/bin/env python
# Copyright (c) Alibaba Group
import argparse
import builtins
import os
import random
import time
import warnings
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import secu.loader
import secu.folder
import secu.builder
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=401, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.2, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--log', type=str)
# options for secu
parser.add_argument('--secu-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--secu-num-ins', default=50000, type=int,
                    help='number of instances (default: 50000)')
parser.add_argument('--secu-num-head', default=10, type=int,
                    help='number of k-means ( default: 10)')
parser.add_argument('--secu-k', default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100], type=int, nargs="+", help='multi-clustering head')
parser.add_argument('--secu-tx', default=0.05, type=float,
                    help='temperature for representation (default: 0.05)')
parser.add_argument('--secu-tw', default=0.05, type=float,
                    help='temperature for cluster center (default: 0.05)')
parser.add_argument('--secu-tau', default=0.2, type=float,
                    help='weight of one-hot label (default: 0.2)')
parser.add_argument('--secu-dual-lr', default=0.1, type=float,
                    help='dual learning rate for lower bound (default: 0.1)')
parser.add_argument('--secu-lratio', default=0.9, type=float,
                    help='lower-bound ratio (default: 0.4)')
parser.add_argument('--secu-alpha', default=6000, type=float,
                    help='entropy weight (default: 6000)')
parser.add_argument('--secu-cst', default='size', type=str,
                    help='constraint in secu: size or entropy')
parser.add_argument('--clr', default=1.2, type=float,
                    help='learning rate for cluster center')
parser.add_argument('--min-crop', default=0.3, type=float,
                    help='minimal scale for random crop')
parser.add_argument('--data-name', default='cifar10', type=str,
                    help='name of data: cifar10, cifar100, stl10')


def main():
    args = parser.parse_args()
    print(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    assert (len(args.secu_k) == args.secu_num_head)
    print("=> creating model")
    if args.data_name == 'stl10':
        from nets.resnet_stl import resnet18
    elif args.data_name=='cifar10' or args.data_name=='cifar100':
        from nets.resnet_cifar import resnet18
    else:
        print("Input data set is not supported")
        return
    model = secu.builder.SeCu(
        base_encoder=resnet18,
        K=args.secu_k,
        tx=args.secu_tx,
        tw=args.secu_tw,
        dim=args.secu_dim,
        num_ins=args.secu_num_ins,
        alpha=args.secu_alpha,
        dual_lr=args.secu_dual_lr,
        lratio=args.secu_lratio,
        constraint=args.secu_cst
    )
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    centers = []
    encoder = []
    for name, param in model.named_parameters():
        if 'center' in name:
            centers.append(param)
        else:
            encoder.append(param)

    optimizer = torch.optim.SGD([{"params": encoder, "lr": args.lr},
                                 {"params": centers, "lr": args.clr}],
                                 weight_decay=args.weight_decay,
                                 momentum=args.momentum)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    model.module.load_param()
    cudnn.benchmark = True
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    if args.data_name == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
        crop_size = 32
    elif args.data_name == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
        crop_size = 32
    elif 'stl' in args.data_name:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        crop_size = 96

    aug_1 = [
        transforms.RandomResizedCrop(crop_size, scale=(args.min_crop, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([secu.loader.GaussianBlur([.1, 2.])], p=1.0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    aug_2 = [
        transforms.RandomResizedCrop(crop_size, scale=(args.min_crop, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([secu.loader.GaussianBlur([.1, 2.])], p=0.1),
        transforms.RandomApply([secu.loader.Solarize()], p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    train_dataset = secu.folder.ImageFolder(
        traindir,
        secu.loader.DoubleCropsTransform(transforms.Compose(aug_1),
                                         transforms.Compose(aug_2)))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=False)

    scaler = GradScaler()
    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, scaler)
        if args.secu_cst == 'size':
            model.module.reset_count()
        print('use time :', time.time() - start_time)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename='model/{}_{:04d}.pth.tar'.format(args.log, epoch))


def train(train_loader, model, criterion, optimizer, epoch, args, scaler):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()
    train_loader_len = len(train_loader)
    pcenters = model.module.get_centers()
    for i, (images, target) in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, args, i, train_loader_len)
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu)
        with autocast():
            loss_x, loss_c = model(images[0], images[1], pcenters, target, epoch, criterion, args)
            loss = loss_x + loss_c
        losses.update(loss.item(), images[0].size(0))
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)
    progress.display(train_loader_len)
    for i in range(0, args.secu_num_head):
        print('max and min cluster size for {}-class clustering is ({},{})'.format(args.secu_k[i], torch.max(
            model.module.counters[i].data).item(), torch.min(model.module.counters[i].data).item()))

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    if (state['epoch'] - 1) % 200 != 0 or state['epoch'] == 1:
        return
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def adjust_learning_rate(optimizer, epoch, args, iteration, num_iter):
    warmup_epoch = 11
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = args.epochs * num_iter
    lr = args.lr * (1 + math.cos(math.pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
    if epoch < warmup_epoch:
        if epoch == 0:
            lr = 0
        else:
            lr = args.lr * max(1, current_iter - num_iter) / (warmup_iter - num_iter)
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = args.clr


if __name__ == '__main__':
    main()

