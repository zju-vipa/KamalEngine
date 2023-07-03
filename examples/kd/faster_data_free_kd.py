
import os
from kamal.utils import set_mode, move_to_device
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from kamal import vision, engine, utils, amalgamation, metrics, callbacks, slim
import kamal
import time
from kamal.core import tasks
from kamal.vision.models import generator
from kamal.vision import sync_transforms as sT
import argparse
from math import gamma
import os
import random
import shutil
import warnings
from tqdm import tqdm
# import datafree
from kamal.slim.distillation import data_free
from kamal.slim.distillation.data_free import criterions
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

parser = argparse.ArgumentParser(
    description='Data-free Knowledge Distillation')

# Data Free
parser.add_argument('--method',
                    required=True,
                    choices=[
                        'zskt', 'dfad', 'dafl', 'deepinv', 'dfq', 'cmi',
                        'fast', 'fast_meta'
                    ])
parser.add_argument('--adv',
                    default=0,
                    type=float,
                    help='scaling factor for adversarial distillation')
parser.add_argument('--bn',
                    default=0,
                    type=float,
                    help='scaling factor for BN regularization')
parser.add_argument('--oh',
                    default=0,
                    type=float,
                    help='scaling factor for one hot loss (cross entropy)')
parser.add_argument('--act',
                    default=0,
                    type=float,
                    help='scaling factor for activation loss used in DAFL')
parser.add_argument('--balance',
                    default=0,
                    type=float,
                    help='scaling factor for class balance')
parser.add_argument('--save_dir', default='run/synthesis', type=str)

parser.add_argument('--cr',
                    default=1,
                    type=float,
                    help='scaling factor for contrastive model inversion')
parser.add_argument('--cr_T',
                    default=0.5,
                    type=float,
                    help='temperature for contrastive model inversion')
parser.add_argument('--cmi_init',
                    default=None,
                    type=str,
                    help='path to pre-inverted data')

parser.add_argument('--lr_g',
                    default=1e-3,
                    type=float,
                    help='initial learning rate for generator')
parser.add_argument('--lr_z',
                    default=1e-3,
                    type=float,
                    help='initial learning rate for latent code')
parser.add_argument('--g_steps',
                    default=1,
                    type=int,
                    metavar='N',
                    help='number of iterations for generation')
parser.add_argument('--reset_l0',
                    default=0,
                    type=int,
                    help='reset l0 in the generator during training')
parser.add_argument('--reset_bn',
                    default=0,
                    type=int,
                    help='reset bn layers during training')
parser.add_argument('--bn_mmt',
                    default=0,
                    type=float,
                    help='momentum when fitting batchnorm statistics')
parser.add_argument('--is_maml',
                    default=1,
                    type=int,
                    help='meta gradient: is maml or reptile')

# Basic
parser.add_argument('--teacher_ckpt', required=True)
parser.add_argument('--data_root', default='./data')
parser.add_argument('--teacher', default='wrn40_2')
parser.add_argument('--student', default='wrn16_1')
parser.add_argument('--dataset', default='cifar100')
parser.add_argument('--lr',
                    default=0.1,
                    type=float,
                    help='initial learning rate for KD')
parser.add_argument('--T', default=1, type=float)

parser.add_argument('--epochs',
                    default=200,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--kd_steps',
                    default=400,
                    type=int,
                    metavar='N',
                    help='number of iterations for KD after generation')
parser.add_argument('--ep_steps',
                    default=400,
                    type=int,
                    metavar='N',
                    help='number of total iterations in each epoch')
parser.add_argument('--warmup',
                    default=0,
                    type=int,
                    metavar='N',
                    help='which epoch to start kd')

parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate_only',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--batch_size',
                    default=128,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                    'batch size of all GPUs on the current node when '
                    'using Data Parallel or Distributed Data Parallel')
parser.add_argument(
    '--synthesis_batch_size',
    default=None,
    type=int,
    metavar='N',
    help='mini-batch size (default: None) for synthesis, this is the total '
    'batch size of all GPUs on the current node when '
    'using Data Parallel or Distributed Data Parallel')

# Device
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
# TODO: Distributed and FP-16 training
parser.add_argument('--world_size',
                    default=-1,
                    type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank',
                    default=-1,
                    type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist_url',
                    default='tcp://224.66.41.62:23456',
                    type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist_backend',
                    default='nccl',
                    type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing_distributed',
                    action='store_true',
                    help='Use multi-processing distributed training to launch '
                    'N processes per node, which has N GPUs. This is the '
                    'fastest way to use PyTorch for either single node or '
                    'multi node data parallel training')
parser.add_argument('--fp16', action='store_true', help='use fp16')

# Misc
parser.add_argument('--seed',
                    default=None,
                    type=int,
                    help='seed for initializing training.')
parser.add_argument('--log_tag', default='')
parser.add_argument('-j',
                    '--workers',
                    default=4,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    metavar='M',
                    help='momentum')
parser.add_argument('--wd',
                    '--weight_decay',
                    default=1e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p',
                    '--print_freq',
                    default=0,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--pretrained',
                    dest='pretrained',
                    action='store_true',
                    help='use pre-trained model')

best_acc1 = 0
time_cost = 0
class Evaluator(object):
    def __init__(self, metric, dataloader):
        self.dataloader = dataloader
        self.metric = metric

    def eval(self, model, device=None, progress=False):
        self.metric.reset()
        with torch.no_grad():
            for i, (inputs, targets) in enumerate( tqdm(self.dataloader, disable=not progress) ):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model( inputs )
                self.metric.update(outputs, targets)
        return self.metric.get_results()
    
    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)

def classification_evaluator(dataloader):
    metric = metrics.MetricCompose({
        'Acc': metrics.TopkAccuracy(),
        'Loss': metrics.RunningLoss(torch.nn.CrossEntropyLoss(reduction='sum'))
    })
    return Evaluator( metric, dataloader=dataloader)

def main():
    args = parser.parse_args()
    global best_acc1
    global time_cost
    ############################################
    # Logger
    ############################################
    if args.log_tag != '':
        args.log_tag = '-'+args.log_tag
    log_name = 'R%d-%s-%s-%s%s'%(args.rank, args.dataset, args.teacher, args.student, args.log_tag) if args.multiprocessing_distributed else '%s-%s-%s'%(args.dataset, args.teacher, args.student)
    args.logger = utils.logger.get_logger(log_name, output='checkpoints/datafree-%s/log-%s-%s-%s%s.txt'%(args.method, args.dataset, args.teacher, args.student, args.log_tag))
    ############################################
    # Setup dataset
    ############################################
    # num_classes, ori_dataset, val_dataset = registry.get_dataset(name=args.dataset, data_root=args.data_root)
    ori_dataset = vision.datasets.torchvision_datasets.CIFAR100(
        './data/torchdata',
        train=True,
        download=True,
        transform=sT.Compose([
            sT.RandomCrop(32, padding=4),
            sT.RandomHorizontalFlip(),
            sT.ToTensor(),
            sT.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2023, 0.1994, 0.2010))
        ]))
    val_dataset = vision.datasets.torchvision_datasets.CIFAR100(
        './data/torchdata',
        train=False,
        download=True,
        transform=sT.Compose([
            sT.ToTensor(),
            sT.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2023, 0.1994, 0.2010))
        ]))
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    evaluator = classification_evaluator(val_loader)
    metric = kamal.tasks.StandardMetrics.classification()
    
    student = vision.models.classification.cifar.wrn.wrn_16_1(num_classes=100)
    teacher = vision.models.classification.cifar.wrn.wrn_40_2(num_classes=100)
    args.normalizer = normalizer = utils.Normalizer(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
    teacher.load_state_dict(torch.load(args.teacher_ckpt))
    criterion =criterions.KLDiv(T=args.T)
    ############################################
    # Setup the data-free synthesizer
    ############################################
    if args.synthesis_batch_size is None:
        args.synthesis_batch_size = args.batch_size
    #不同的生成对抗样本的方式
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    student, teacher = student.to(device), teacher.to(device)
    global generator
    if args.method == 'deepinv':  
        synthesizer = data_free.DeepInvSyntheiszer(
            teacher=teacher,
            student=student,
            num_classes=100,
            img_size=(3, 32, 32),
            iterations=args.g_steps,
            lr_g=args.lr_g,
            synthesis_batch_size=args.synthesis_batch_size,
            sample_batch_size=args.batch_size,
            adv=args.adv,
            bn=args.bn,
            oh=args.oh,
            tv=0.0,
            l2=0.0,
            save_dir=args.save_dir,
            transform=ori_dataset.transform,
            normalizer=args.normalizer,
            device=device)
    elif args.method in ['zskt', 'dfad', 'dfq', 'dafl']:
        nz = 512 if args.method == 'dafl' else 256
        generator = generator.Generator(nz=nz, ngf=64, img_size=32, nc=3)
        generator = generator.to(device)
        criterion = torch.nn.L1Loss() if args.method == 'dfad' else criterions.KLDiv()
        synthesizer = data_free.GenerativeSynthesizer(
            teacher=teacher,
            student=teacher,
            generator=generator,
            nz=nz,
            img_size=(3, 32, 32),
            iterations=args.g_steps,
            lr_g=args.lr_g,
            synthesis_batch_size=args.synthesis_batch_size,
            sample_batch_size=args.batch_size,
            adv=args.adv,
            bn=args.bn,
            oh=args.oh,
            act=args.act,
            balance=args.balance,
            criterion=criterion,
            normalizer=args.normalizer,
            device=device)
    elif args.method == 'cmi':
        nz = 256
        generator = generator.Generator(nz=nz, ngf=64, img_size=32, nc=3)
        generator = generator.to(device)
        feature_layers = None  # use outputs from all conv layers
        if args.teacher == 'resnet34':  # use block outputs
            feature_layers = [
                teacher.layer1, teacher.layer2, teacher.layer3, teacher.layer4
            ]
        synthesizer = data_free.CMISynthesizer(
            teacher,
            student,
            generator,
            nz=nz,
            num_classes=100,
            img_size=(3, 32, 32),
            feature_reuse=False,
            # if feature layers==None, all convolutional layers will be used by CMI.
            feature_layers=feature_layers,
            bank_size=40960,
            n_neg=4096,
            head_dim=256,
            init_dataset=args.cmi_init,
            iterations=args.g_steps,
            lr_g=args.lr_g,
            progressive_scale=False,
            synthesis_batch_size=args.synthesis_batch_size,
            sample_batch_size=args.batch_size,
            adv=args.adv,
            bn=args.bn,
            oh=args.oh,
            cr=args.cr,
            cr_T=args.cr_T,
            save_dir=args.save_dir,
            transform=ori_dataset.transform,
            normalizer=args.normalizer,
            device=device)
    elif args.method == 'fast':
        nz = 256
        generator = generator.Generator(nz=nz, ngf=64, img_size=32, nc=3)
        generator = generator.to(device)
        synthesizer = data_free.FastSynthesizer(
            teacher,
            student,
            generator,
            nz=nz,
            num_classes=100,
            img_size=(3, 32, 32),
            init_dataset=args.cmi_init,
            save_dir=args.save_dir,
            device=device,
            transform=ori_dataset.transform,
            normalizer=args.normalizer,
            synthesis_batch_size=args.synthesis_batch_size,
            sample_batch_size=args.batch_size,
            iterations=args.g_steps,
            warmup=args.warmup,
            lr_g=args.lr_g,
            lr_z=args.lr_z,
            adv=args.adv,
            bn=args.bn,
            oh=args.oh,
            reset_l0=args.reset_l0,
            reset_bn=args.reset_bn,
            bn_mmt=args.bn_mmt,
            is_maml=args.is_maml)
    elif args.method == 'fast_meta':
        nz = 256
        generator = generator.Generator(nz=nz, ngf=64, img_size=32, nc=3)
        generator = generator.to(device)
        synthesizer = data_free.FastMetaSynthesizer(
            teacher,
            student,
            generator,
            nz=nz,
            num_classes=100,
            img_size=(3, 32, 32),
            init_dataset=args.cmi_init,
            save_dir=args.save_dir,
            device=device,
            transform=ori_dataset.transform,
            normalizer=args.normalizer,
            synthesis_batch_size=args.synthesis_batch_size,
            sample_batch_size=args.batch_size,
            iterations=args.g_steps,
            warmup=args.warmup,
            lr_g=args.lr_g,
            lr_z=args.lr_z,
            adv=args.adv,
            bn=args.bn,
            oh=args.oh,
            reset_l0=args.reset_l0,
            reset_bn=args.reset_bn,
            bn_mmt=args.bn_mmt,
            is_maml=args.is_maml)
    else:
        raise NotImplementedError
    ############################################
    # Setup optimizer
    ############################################
    optimizer = torch.optim.SGD(student.parameters(),
                                args.lr,
                                weight_decay=args.weight_decay,
                                momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           200,
                                                           eta_min=2e-4)
# do the distillation
    logger = utils.logger.get_logger('fkd_%s' % (args.dataset))
    tb_writer = SummaryWriter(log_dir='run/fkd%s-%s' %
                              (args.dataset, time.asctime().replace(' ', '_')))
    
    distiller = slim.FKDDistiller(logger, tb_writer)
    distiller.setup(student, teacher, scheduler,evaluator,synthesizer, val_loader, optimizer, criterion ,args, device)
    
    distiller.run(start_iter=0, max_iter=len(val_loader)*args.epochs)

if __name__ == '__main__':
    main()