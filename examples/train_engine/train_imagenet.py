import argparse
import os, sys
import torch
import torch.nn as nn

from PIL import Image

from kamal import engine, metrics
from kamal.vision.models.classification.darknet import darknet19, darknet53
from kamal.vision.datasets import ImageNet
from kamal.vision import sync_transforms as transforms
from visdom import Visdom
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--data_root', type=str, default='./data/ILSVRC2012')
    parser.add_argument('--model', type=str, required=True, choices=['darknet19', 'darknet53'])
    args = parser.parse_args()

    train_loader = torch.utils.data.DataLoader(
        ImageNet(args.data_root, split='train', transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
        ), batch_size=64, num_workers=4, shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        ImageNet(args.data_root, split='val', transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])]) 
        ), batch_size=64, num_workers=4
    )

    task = engine.task.ClassificationTask( criterion=nn.CrossEntropyLoss(ignore_index=255) )
    if args.model=='darknet19':
        model = darknet19(num_classes=1000)
    elif args.model=='darknet53':
        model = darknet53(num_classes=1000)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(train_loader)*30, gamma=0.1)
    trainer = engine.trainer.SimpleTrainer( task, model, train_loader, optimizer, logs_fname='%s-imagenet'%args.model)

    vis = Visdom(port='29999', env=args.model)
    trainer.add_callbacks( [
        engine.callbacks.LoggingCallback(
            interval=100, 
            vis=vis, 
            name=('total_loss',), 
            smooth_window_size=100),
        engine.callbacks.LRSchedulerCallback(scheduler),
        engine.callbacks.SimpleValidationCallback(
            interval=5000, 
            val_loader=val_loader, 
            metrics=metrics.StreamClassificationMetrics(),
            save_model=('best', 'latest'), 
            ckpt_dir='checkpoints',
            ckpt_name=args.model,
            vis = vis)  
    ] )
    trainer.train(0, len(train_loader)*90)

if __name__=='__main__':
    main()