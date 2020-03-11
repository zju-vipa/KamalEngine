import argparse
import os, sys
import torch
import torch.nn as nn

from PIL import Image

from kamal import engine, metrics
from kamal.vision.models.classification import resnet18
from torchvision.datasets import CIFAR10
from torchvision import transforms as T
from visdom import Visdom
import random
 
# 1. Run visdom server: $ visdom -p 29999
# 2. python train_camvid.py --lr 0.01


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--data_root', type=str, default='./data')
    args = parser.parse_args()

    train_loader = torch.utils.data.DataLoader(
        CIFAR10(args.data_root, train=True, download=True, 
            transform=T.Compose([
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])])
            ), batch_size=128, num_workers=4, shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        CIFAR10(args.data_root, train=False, download=True, 
            transform=T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])]) 
            ), batch_size=128, num_workers=4
    )
    
    # Prepare model
    teacher = resnet18()
    teacher.fc = nn.Linear( teacher.fc.in_features, 10 )
    teacher.load_state_dict( torch.load( 'checkpoints/cifar10-best-00025806-acc-0.863.pth') )

    student = resnet18()
    student.fc = nn.Linear( student.fc.in_features, 10 )

    # prepare trainer
    viz = Visdom(port='29999', env='cifar10-kd')
    task = engine.task.KDClassificationTask()
    optimizer = torch.optim.SGD(student.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*100)
    evaluator = engine.evaluator.ClassificationEvaluator( val_loader )
    trainer = engine.trainer.SimpleKDTrainer( task=task, model=student, teacher=teacher, viz=viz )

    trainer.add_callbacks( [
        engine.callbacks.LoggingCallback(
            interval=50,  
            names=('total_loss', 'lr'), 
            smooth_window_sizes=( 20, None )
        ),
        engine.callbacks.LRSchedulerCallback(scheduler=scheduler),
        engine.callbacks.ValidationCallback(
            interval=len(train_loader), 
            evaluator=evaluator, 
            save_model=('best', 'latest'), 
            ckpt_dir='checkpoints',
            ckpt_tag='cifar10'
        )
    ] )
    trainer.train(0, len(train_loader)*100, train_loader=train_loader, optimizer=optimizer)

if __name__=='__main__':
    main()