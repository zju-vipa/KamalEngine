import pickle

import torch, time, os
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


import random

import warnings

from kamal import vision, amalgamation, engine, utils, metrics, callbacks ,tasks
from kamal.core.engine import split_cifar100
# from kamal.amalgamation import OOD_KA_amal
from kamal.vision import sync_transforms as sT
from kamal.vision.models import generator
from tqdm import tqdm
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import argparse

parser = argparse.ArgumentParser()
# model & dataset
parser.add_argument('--data_root', default='data')
parser.add_argument('--teacher0_ckpt', required=True )
parser.add_argument('--teacher1_ckpt', required=True )
parser.add_argument('--model', default='wrn16_2')
parser.add_argument('--dataset', default='cifar100')
parser.add_argument('--unlabeled', default='cifar10')
# train detail
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')
parser.add_argument('--iter', type=int, default=0)
parser.add_argument('-b', '--batch_size', default=256, type=int)
parser.add_argument('--lr', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--lr_g', default=1e-3, type=float)
parser.add_argument('--z_dim', default=100, type=int)
parser.add_argument('-k', '--k_step', default=5, type=int)
# loss weight
parser.add_argument('--oh', default=1.0, type=float)
parser.add_argument('--bn', default=1.0, type=float)
parser.add_argument('--local', default=1.0, type=float)
parser.add_argument('--adv', default=1.0, type=float)
parser.add_argument('--sim', default=1.0, type=float)
parser.add_argument('--balance', default=10.0, type=float)
parser.add_argument('--kd', default=1.0, type=float)
parser.add_argument('--amal', default=1.0, type=float)
parser.add_argument('--recons', default=1.0, type=float)

parser.add_argument('--test_only', action='store_true', default=False)
parser.add_argument('--ckpt', type=str, default=None)
args = parser.parse_args()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ==================================================
    # ==================== Dataset =====================
    # ==================================================
    train_transform = sT.Compose([
            sT.RandomCrop(32, padding=4),
            sT.RandomHorizontalFlip(),
            sT.ToTensor(),
            sT.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
    val_transform = sT.Compose([
            sT.ToTensor(),
            sT.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
    part0_train = split_cifar100.CIFAR100_PART('../data/torchdata', train=True, part = 0, transform=train_transform)
    part0_val = split_cifar100.CIFAR100_PART('../data/torchdata', train=False, part = 0, transform=val_transform)
    part1_train = split_cifar100.CIFAR100_PART('../data/torchdata', train=True, part = 1, transform=train_transform)
    part1_val = split_cifar100.CIFAR100_PART('../data/torchdata', train=False, part = 1, transform=val_transform)
    # ==================================================
    # ===================== Model ======================
    # ==================================================
    part0_teacher = vision.models.classification.cifar.wrn.wrn_40_2(num_classes=50)
    part1_teacher = vision.models.classification.cifar.wrn.wrn_40_2(num_classes=50)
    student = vision.models.classification.cifar.wrn.wrn_40_2(num_classes=100)
    
    netG = generator.Generator(nz=args.z_dim, nc=3, img_size=32)
    netD = generator.PatchDiscriminator(nc=3, ndf=128)

    part0_teacher.load_state_dict( torch.load( args.teacher0_ckpt ) )
    part1_teacher.load_state_dict( torch.load( args.teacher1_ckpt ) )
    # ==================================================
    # ================== OOD Dataset ===================
    # ==================================================
    normalizer = utils._utils.Normalizer(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
    args.normalizer = normalizer
    val_dataset = vision.datasets.torchvision_datasets.CIFAR100( 
        '../data/torchdata', train=False, download=True, transform=sT.Compose([
            sT.ToTensor(),
            sT.Normalize( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) )
        ]) )
    ood_with_aug = vision.datasets.torchvision_datasets.CIFAR10( 
        '../data/torch10', train=True, download=True, transform=sT.Compose([
            sT.RandomCrop(32, padding=4),
            sT.RandomHorizontalFlip(),
            sT.ToTensor(),
            sT.Normalize( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) )
        ]) )    
    ood_without_aug = vision.datasets.torchvision_datasets.CIFAR10( 
        '../data/torch10', train=True, download=True, transform=sT.Compose([
            sT.RandomCrop(32, padding=4),
            sT.RandomHorizontalFlip(),
            sT.ToTensor(),
            sT.Normalize( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) )
        ]) )  

    ood_with_aug.transforms = ood_with_aug.transform = part0_train.transform  # with aug
    ood_without_aug.transforms = ood_without_aug.transform = part0_val.transform  # without aug

    if args.unlabeled in ['imagenet_32x32', 'places365_32x32']:
        ood_index_root = os.path.join(args.data_root, 'ood_index_%s_%s.pkl' % (args.unlabeled, args.model))

        if not os.path.exists(ood_index_root):
            ood_index = utils._utils.prepare_ood_subset(ood_without_aug, 50000,
                                           nn.ModuleList([part0_teacher, part1_teacher]).to(device))

            with open(ood_index_root, 'wb') as f:
                pickle.dump(ood_index, f)

        with open(ood_index_root, 'rb') as f:
            ood_index = pickle.load(f)
        ood_with_aug.samples = [ood_with_aug.samples[i] for i in ood_index]
        ood_without_aug.samples = [ood_without_aug.samples[i] for i in ood_index]

    # ==================================================
    # =================== DataLoader ===================
    # ==================================================
    ood_with_aug_loader = torch.utils.data.DataLoader(ood_with_aug, batch_size=args.batch_size, shuffle=True,
                                                      num_workers=4)
    ood_without_aug_loader = torch.utils.data.DataLoader(ood_without_aug, batch_size=args.batch_size, shuffle=True,
                                                         num_workers=4)
    part0_val_loader = torch.utils.data.DataLoader(part0_val, batch_size=args.batch_size, shuffle=False, num_workers=4)
    part1_val_loader = torch.utils.data.DataLoader(part1_val, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # ==================================================
    # =================== Optimizer ====================
    # ==================================================
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    TOTAL_ITERS = len(ood_with_aug_loader) * args.epochs
    optim_s = torch.optim.Adam(student.parameters(), lr=args.lr, weight_decay=1e-4)
    sched_s = torch.optim.lr_scheduler.CosineAnnealingLR(optim_s, T_max=TOTAL_ITERS)
    optim_g = torch.optim.Adam(netG.parameters(), lr=args.lr_g, betas=[0.5, 0.999])
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(optim_g, T_max=TOTAL_ITERS)
    optim_d = torch.optim.Adam(netD.parameters(), lr=args.lr_g, betas=[0.5, 0.999])
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(optim_d, T_max=TOTAL_ITERS)
    metric = tasks.StandardMetrics.classification()
    val_evaluator = engine.evaluator.BasicEvaluator( val_loader, metric )
    # ==================================================
    # ==================== Trainer =====================
    # ==================================================
    output_dir = 'run/OOD_KA_%s' % (time.asctime().replace(' ', '_'))
    trainer = amalgamation.OOD_KA_Amalgamator(
        logger=utils.logger.get_logger(name='OOD-KA', output=os.path.join(output_dir, 'log.txt')),
        tb_writer=SummaryWriter( log_dir='run/OOD_KA-%s'%( time.asctime().replace( ' ', '_' ) ) ) 
    )
    # #     trainer = amalgamation.LayerWiseAmalgamator( 
    # #     logger=utils.logger.get_logger('layerwise-ka'), 
    # #     tb_writer=SummaryWriter( log_dir='run/layerwise_ka-%s'%( time.asctime().replace( ' ', '_' ) ) ) 
    # # )
    # # for k, v in flatten_dict(vars(args)).items():  # print args
    # #     trainer.logger.info("%s: %s" % (k, v))
    # trainer.add_callback( 
    #     engine.DefaultEvents.AFTER_STEP(every=10), 
    #     callbacks=callbacks.MetricsLogging(keys=('loss_ka', 'loss_kd', 'loss_amal', 'loss_recons', 'lr')))
    # trainer.add_callback( 
    #     engine.DefaultEvents.AFTER_EPOCH, 
    #     callbacks=[
    #         callbacks.EvalAndCkpt(model=student, evaluator=val_evaluator, metric_name='acc', ckpt_prefix='MosaicKD'),
    #     ] )
    # #EvalAndCkpt可保存模型路径
    # # 添加 AFTER_STEP 事件的回调函数，学习率调整,学习率调度回调函数，需要指定学习率调度器schedulers。
    # trainer.add_callback(
    #     engine.DefaultEvents.AFTER_STEP,
    #     callbacks=callbacks.LRSchedulerCallback(schedulers=[sched_s, sched_g, sched_d]))
    trainer.setup(args=args,
                  student=student,
                  teachers=[part0_teacher, part1_teacher],
                  netG=netG,
                  netD=netD,
                  train_loader=[ood_with_aug_loader, ood_without_aug_loader],
                  val_loaders=[part0_val_loader, part1_val_loader],
                  val_num_classes=[50, 50],
                  optimizers=[optim_s, optim_g, optim_d],
                  schedulers=[sched_s, sched_g, sched_d],
                  device=device)

    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt)
        
        trainer.student.load_state_dict(checkpoint['s_state_dict'])
        trainer.args.iter = checkpoint['iter']
        print(trainer.args.iter)
        trainer.netG.load_state_dict(checkpoint['g_state_dict'])
        trainer.netD.load_state_dict(checkpoint['d_state_dict'])
        print("Load student model from %s" % args.ckpt)
    if args.test_only:
        trainer.validate()
        return

    trainer.run(start_iter=0, max_iter=TOTAL_ITERS)


if __name__ == '__main__':
    main()