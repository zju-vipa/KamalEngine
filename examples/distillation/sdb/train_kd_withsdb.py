import sys
sys.path.append("../../../")

import torch, time
import torch.nn as nn
from torch.optim import SGD, Adam

from tqdm import tqdm
import argparse
import os
import logging
import numpy as np

import kamal
from kamal.vision import sync_transforms as sT
from torch.utils.tensorboard import SummaryWriter
from kamal import vision, engine, callbacks
from kamal.distillation.sdb.safe_distillation_box import AdversTeacher,AdversTEvaluator,KD_SDB_Stuednt
from kamal.distillation.sdb.sdb_task import SDBTask,KD_SDB_Task

# random seed
seed = 0

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# args 
parser = argparse.ArgumentParser()
parser.add_argument('--noise_path', default='./CIFAR10/', type=str)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--stu_model_name', default='resnet18', type=str)
parser.add_argument('--teacher_model', default='resnet18', type=str)
parser.add_argument('--teacher_resume', default='./checkpoints/cifar10_SDBtch_acc_0.8652.pth', type=str)

parser.add_argument('--learning_rate', default=0.1, type=float)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--temperature', default=4, type=float)
parser.add_argument('--weight', default=1, type=float)
parser.add_argument('--lamb', default=0.9, type=float)
parser.add_argument('--eta', default=0.01, type=float)
parser.add_argument('--alpha', default=1, type=float)
parser.add_argument('--cuda', default=True, type=bool)
args = parser.parse_args()

device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

def generate_noise(params):
    data_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    if params.dataset == 'cifar10' or params.dataset == 'cifar100':
        random.seed(1)
        noisy_img = np.random.randint(0, 255, size=(32, 32, 3))
        return data_transformer(noisy_img / 255.0).float()
    else:
        noisy_img = np.random.randint(0, 255, size=(64, 64, 3))
        return data_transformer(noisy_img / 255.0).float()

def main(args): 
    # Dataset
    train_dst = vision.datasets.torchvision_datasets.CIFAR10( 
        'data/data-cifar10', train=True, download=True, transform=sT.Compose([
            sT.RandomCrop(32, padding=4),
            sT.RandomHorizontalFlip(),
            sT.ToTensor(),
            sT.Normalize( mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261) )
        ]) )
    val_dst = vision.datasets.torchvision_datasets.CIFAR10( 
        'data/data-cifar10', train=False, download=True, transform=sT.Compose([
            sT.ToTensor(),
            sT.Normalize( mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261) )
        ]) )
    train_loader = torch.utils.data.DataLoader( train_dst, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers )
    val_loader = torch.utils.data.DataLoader( val_dst, batch_size=args.batch_size, num_workers=args.num_workers )

     # Model 
    if args.dataset == 'cifar10':
        num_class = 10

    print('Number of class: ' + str(num_class))
    print('Create Student Model --- ' + args.stu_model_name)

    if args.stu_model_name == 'resnet18':
        model = vision.models.classification.resnet18(num_classes=num_class)
    else:
        model = None
        print('Not support for model ' + str(args.stu_model_name))
        exit()

    # Teacher Model
    if args.teacher_model == 'resnet18':
        teacher_model = vision.models.classification.cifar.wrn.wrn_40_2(num_classes=num_class)
    else:
        teacher_model = None
        print('Not support for model ' + str(args.teacher_model))
        exit()

    if args.cuda:
        model = model.cuda()
        teacher_model = teacher_model.cuda()

    # checkpoint
    if args.resume:
        print('- Load checkpoint model from {}'.format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint)
    else:
        print('- Train from scratch ')

    # load teacher model
    if args.teacher_resume:
        teacher_resume = args.teacher_resume
        print('------ Teacher Resume from {}'.format(teacher_resume))
    else:
        print('Please Load a Trained Teacher!')
    checkpoint = torch.load(teacher_resume)
    teacher_model.load_state_dict(checkpoint)

    # Optimizer
    if args.stu_model_name == 'CNN' or args.stu_model_name == 'mlp':
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
        print('Optimizer: Adam')
    else:
        optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
        print('Optimizer: SGD')
    TOTAL_ITERS= len(train_loader) * args.num_epochs
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=TOTAL_ITERS )

    # KAE Part
    # prepare noise
    if (os.path.exists(os.path.join(args.noise_path, 'noise.pth'))):
        noise_data = torch.load(os.path.join(args.noise_path, 'noise.pth'))
        print("Use noise_data resume {}".format(args.noise_path)+'noise.pth')
    else:
        noise_data = generate_noise(args)
        torch.save(noise_data, os.path.join(args.noise_path, 'noise.pth'))
        print("Create noise_data")
        
    if args.cuda:
        noise_data = noise_data.cuda()

    # prepare evaluator
    metric = kamal.tasks.StandardMetrics.classification()
    evaluator = engine.evaluator.BasicEvaluator(dataloader=val_loader, metric=metric, progress=False)

    # prepare trainer
    task = KD_SDB_Task(name='KD_SDB_Teacher',
            loss_fn_kd=nn.KLDivLoss(reduction='batchmean'),
            loss_fn_ce=nn.CrossEntropyLoss(),
            scaling=1.0, 
            pred_fn=lambda x: x.max(1)[1], 
            attach_to=None
    )
    trainer = KD_SDB_Stuednt( 
        logger=kamal.utils.logger.get_logger('cifar10'), 
        tb_writer=SummaryWriter( log_dir='run/cifar10-%s'%( time.asctime().replace( ' ', '_' ) ) ) 
    )
    trainer.setup( student=model, 
                teacher=teacher_model,
                task=task, 
                dataloader=train_loader,
                optimizer=optimizer,
                params=args,
                noise=noise_data,
                device=device)

    # add callbacks
    trainer.add_callback( 
        engine.DefaultEvents.AFTER_EPOCH, 
        callbacks=callbacks.EvalAndCkpt(model=model, evaluator=evaluator, metric_name='acc', ckpt_prefix='cifar10_SDB_KD_student') )
    trainer.add_callback(
        engine.DefaultEvents.AFTER_STEP,
        callbacks=callbacks.LRSchedulerCallback(schedulers=[sched]))
    # run
    trainer.run(start_iter=0, max_iter=TOTAL_ITERS)

if __name__=='__main__':
    main(args)