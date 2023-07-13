import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))

import argparse
import random
import warnings

import registry

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import time
from torch.utils.tensorboard import SummaryWriter

from PIL import PngImagePlugin

from kamal.slim.distillation.mosaic_kd import MosaicKD
from kamal.utils import flatten_dict, Normalizer, get_logger, dummy_ctx
from kamal.vision.models.generator import Generator, PatchDiscriminator

LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024 ** 2)

parser = argparse.ArgumentParser(description='MosaicKD for OOD data')
parser.add_argument('--data_root', default='/nfs3/lzc')
parser.add_argument('--teacher', default='wrn40_2')
parser.add_argument('--student', default='wrn16_1')
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--unlabeled', default='cifar10')
parser.add_argument('--log_tag', default='')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr_g', default=1e-3, type=float)
parser.add_argument('--T', default=1, type=float)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

parser.add_argument('--z_dim', default=100, type=int)
parser.add_argument('--output_stride', default=1, type=int)
parser.add_argument('--align', default=0.1, type=float)
parser.add_argument('--local', default=0.1, type=float)
parser.add_argument('--adv', default=1.0, type=float)

parser.add_argument('--balance', default=0.0, type=float)

parser.add_argument('-p', '--print-freq', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--fp16', action='store_true',
                    help='use fp16')
parser.add_argument('--ood_subset', action='store_true',
                    help='use ood subset')
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
best_acc1 = 0


def main():
    args = parser.parse_args()
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
    global best_acc1
    args.gpu = gpu
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

    if args.log_tag != '':
        args.log_tag = '-' + args.log_tag
    log_name = '%s-%s-%s' % (
        args.dataset, args.teacher, args.student) if args.multiprocessing_distributed else '%s-%s-%s' % (
        args.dataset, args.teacher, args.student)
    logger = get_logger(log_name, output='checkpoints/MosaicKD/log-%s-%s-%s-%s%s.txt' % (
        args.dataset, args.unlabeled, args.teacher, args.student, args.log_tag))
    tb_writter = SummaryWriter(log_dir=os.path.join('tb_log', log_name + '_%s' % (time.asctime().replace(' ', '-'))))
    if args.rank <= 0:
        for k, v in flatten_dict(vars(args)).items():  # print args
            logger.info("%s: %s" % (k, v))

    ############################################
    # Setup Dataset
    ############################################
    num_classes, ori_dataset, val_dataset = registry.get_dataset(name=args.dataset, data_root=args.data_root)
    _, train_dataset, _ = registry.get_dataset(name=args.unlabeled, data_root=args.data_root)
    _, ood_dataset, _ = registry.get_dataset(name=args.unlabeled, data_root=args.data_root)
    # see Appendix Sec 2, ood data is also used for training
    ood_dataset.transforms = ood_dataset.transform = train_dataset.transform  # w/o augmentation
    train_dataset.transforms = train_dataset.transform = val_dataset.transform  # w/ augmentation

    ############################################
    # Setup Models
    ############################################
    student = registry.get_model(args.student, num_classes=num_classes)
    teacher = registry.get_model(args.teacher, num_classes=num_classes, pretrained=True).eval()
    teacher.load_state_dict(
        torch.load('checkpoints/pretrained/%s_%s.pth' % (args.dataset, args.teacher), map_location='cpu')['state_dict'])
    normalizer = Normalizer(**registry.NORMALIZE_DICT[args.dataset])
    args.normalizer = normalizer
    netG = Generator(nz=args.z_dim, nc=3, img_size=32)
    netD = PatchDiscriminator(nc=3, ndf=128)

    if args.ood_subset and args.unlabeled in ['imagenet_32x32', 'places365_32x32']:
        ood_index = prepare_ood_data(train_dataset, teacher, ood_size=len(ori_dataset), args=args)
        train_dataset.samples = [train_dataset.samples[i] for i in ood_index]
        ood_dataset.samples = [ood_dataset.samples[i] for i in ood_index]

    if args.distributed:
        process_group = torch.distributed.new_group()
        netD = torch.nn.SyncBatchNorm.convert_sync_batchnorm(netD, process_group)

    ############################################
    # Device preparation
    ############################################
    if not torch.cuda.is_available():
        logger.warning('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            student.cuda(args.gpu)
            teacher.cuda(args.gpu)
            netG.cuda(args.gpu)
            netD.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            student = torch.nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
            teacher = torch.nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
            netG = torch.nn.parallel.DistributedDataParallel(netG, device_ids=[args.gpu])
            netD = torch.nn.parallel.DistributedDataParallel(netD, device_ids=[args.gpu])
        else:
            student.cuda()
            teacher.cuda()
            student = torch.nn.parallel.DistributedDataParallel(student)
            teacher = torch.nn.parallel.DistributedDataParallel(teacher)
            netG = torch.nn.parallel.DistributedDataParallel(netG)
            netD = torch.nn.parallel.DistributedDataParallel(netD)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        student = student.cuda(args.gpu)
        teacher = teacher.cuda(args.gpu)
        netG = netG.cuda(args.gpu)
        netD = netD.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        student = torch.nn.DataParallel(student).cuda()
        teacher = torch.nn.DataParallel(teacher).cuda()
        netG = torch.nn.DataParallel(netG).cuda()
        netD = torch.nn.DataParallel(netD).cuda()

    ############################################
    # Setup dataset
    ############################################
    # cudnn.benchmark = False
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, sampler=train_sampler)
    ood_loader = torch.utils.data.DataLoader(
        ood_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)

    ############################################
    # Setup optimizer
    ############################################
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optim_g = torch.optim.Adam(netG.parameters(), lr=args.lr_g, betas=[0.5, 0.999])
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(optim_g, T_max=args.epochs * len(train_loader))
    optim_d = torch.optim.Adam(netD.parameters(), lr=args.lr_g, betas=[0.5, 0.999])
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(optim_d, T_max=args.epochs * len(train_loader))

    optim_s = torch.optim.SGD(student.parameters(), args.lr, momentum=0.9, weight_decay=args.weight_decay)
    sched_s = torch.optim.lr_scheduler.CosineAnnealingLR(optim_s, T_max=args.epochs * len(train_loader))

    ############################################
    # Resume
    ############################################
    args.current_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            try:
                student.module.load_state_dict(checkpoint['s_state_dict'])
                netG.module.load_state_dict(checkpoint['g_state_dict'])
                netD.module.load_state_dict(checkpoint['d_state_dict'])
            except:
                student.load_state_dict(checkpoint['s_state_dict'])
                netG.load_state_dict(checkpoint['g_state_dict'])
                netD.load_state_dict(checkpoint['d_state_dict'])
            best_acc1 = checkpoint['best_acc1']
            try:
                args.start_epoch = checkpoint['epoch']
                optim_g.load_state_dict(checkpoint['optim_g'])
                sched_g.load_state_dict(checkpoint['sched_g'])
                optim_s.load_state_dict(checkpoint['optim_s'])
                sched_s.load_state_dict(checkpoint['sched_s'])
            except:
                print("Fails to load additional model information")
            print("[!] loaded checkpoint '{}' (epoch {} acc {})"
                  .format(args.resume, checkpoint['epoch'], best_acc1))
        else:
            print("[!] no checkpoint found at '{}'".format(args.resume))

    ############################################
    # Evaluate
    ############################################
    # if args.evaluate:
    #     acc1 = validate(0, val_loader, student, criterion, args)
    #     return

    ############################################
    # Train Loop
    ############################################
    if args.fp16:
        from torch.cuda.amp import autocast, GradScaler
        args.scaler_s = GradScaler() if args.fp16 else None
        args.scaler_g = GradScaler() if args.fp16 else None
        args.scaler_d = GradScaler() if args.fp16 else None
        args.autocast = autocast
    else:
        args.autocast = dummy_ctx

    trainer = MosaicKD(logger, tb_writter)
    trainer.setup(
        args,
        student, teacher,
        netG, netD,
        train_loader,
        ood_loader,
        val_loader,
        [optim_s, optim_g, optim_d],
        [sched_s, sched_g, sched_d],
        criterion,
        train_sampler,
        torch.cuda.device_count(),
        args.gpu
    )
    trainer.run(args.epochs, args.start_epoch)


def prepare_ood_data(train_dataset, model, ood_size, args):
    model.eval()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)
    if os.path.exists('checkpoints/ood_index/%s-%s-%s-ood-index.pth' % (args.dataset, args.unlabeled, args.teacher)):
        ood_index = torch.load(
            'checkpoints/ood_index/%s-%s-%s-ood-index.pth' % (args.dataset, args.unlabeled, args.teacher))
    else:
        with torch.no_grad():
            entropy_list = []
            model.cuda(args.gpu)
            model.eval()
            for i, (images, target) in enumerate(tqdm(train_loader)):
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)
                # compute output
                output = model(images)
                p = torch.nn.functional.softmax(output, dim=1)
                ent = -(p * torch.log(p)).sum(dim=1)
                entropy_list.append(ent)
            entropy_list = torch.cat(entropy_list, dim=0)
            ood_index = torch.argsort(entropy_list, descending=True)[:ood_size].cpu().tolist()
            model.cpu()
            os.makedirs('checkpoints/ood_index', exist_ok=True)
            torch.save(ood_index,
                       'checkpoints/ood_index/%s-%s-%s-ood-index.pth' % (args.dataset, args.unlabeled, args.teacher))
    return ood_index


if __name__ == '__main__':
    main()
