import argparse
import os, sys
import torch
import torch.nn as nn

from PIL import Image

from kamal import engine, metrics
from lenet import LeNet5
from torchvision.datasets import MNIST
from torchvision import transforms as T
from visdom import Visdom
import random
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--data_root', type=str, default='~/Datasets/torchdata')
    args = parser.parse_args()

    train_loader = torch.utils.data.DataLoader(
        MNIST(args.data_root, train=True, download=True, 
            transform=T.Compose([
                T.Resize(32),
                T.ToTensor(),
                ])), 
            batch_size=128, num_workers=4
    )

    val_loader = torch.utils.data.DataLoader(
        MNIST(args.data_root, train=False, download=True, 
            transform=T.Compose([
                T.Resize(32),
                T.ToTensor(),
                ])), 
            batch_size=128, num_workers=4
    )
    
    # Prepare model
    task = engine.task.ClassificationTask()
    model = LeNet5()
    
    # prepare trainer
    viz = None #Visdom(port='29999', env='mnist')
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*20)
    evaluator = engine.evaluator.ClassificationEvaluator( val_loader )
    trainer = engine.trainer.SimpleTrainer( task=task, model=model, viz=viz )

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
            ckpt_tag='mnist-lenet'
        )
    ] )
    trainer.train(0, len(train_loader)*20, train_loader=train_loader, optimizer=optimizer)

if __name__=='__main__':
    main()