import sys, os
import cifar_resnet as resnet

from kamal import slim
from kamal.slim import torch_pruning as tp
from kamal import engine

import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn 
import numpy as np 

from torchvision.datasets import CIFAR10
from torchvision import transforms as T

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True, choices=['train', 'prune', 'test'])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--total_epochs', type=int, default=100)
parser.add_argument('--step_size', type=int, default=70)
parser.add_argument('--round', type=int, default=1)

args = parser.parse_args()

def get_dataloader():
    train_loader = torch.utils.data.DataLoader(
        CIFAR10('../data/torchdata', train=True, transform=T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ]), download=True),batch_size=args.batch_size, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        CIFAR10('../data/torchdata', train=False, transform=T.Compose([
            T.ToTensor(),
        ]),download=True),batch_size=args.batch_size, num_workers=2)
    return train_loader, test_loader

def prune_model(model):
    prunner = slim.Pruner( slim.prunning.LNStrategy(n=1) )
    model = prunner.prune( model, rate=0.2, example_inputs=torch.randn(1, 3, 32, 32) )
    return model    

def train_model(model, epochs=100, lr=0.1, weights_only=True):
    train_loader, test_loader = get_dataloader()
    trainer = engine.trainer.BasicTrainer()
    task = engine.task.ClassificationTask()

    model = resnet.ResNet18(num_classes=10)
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR( optim, T_max=epochs*len(train_loader) )
    evaluator = engine.evaluator.ClassificationEvaluator(test_loader)
    trainer.setup(model, task, data_loader=train_loader, optimizer=optim)
    trainer.add_callbacks([
        engine.callbacks.LRSchedulerCallback( scheduler=[sched] ),
        engine.callbacks.ValidationCallback(
            interval=len(train_loader), 
            evaluator=evaluator, 
            weights_only=weights_only,
            save_type=('best', ),
            ckpt_tag='round-%s'%args.round)
    ])
    trainer.run( 0, epochs*len(train_loader) )
    trainer.callbacks[1].final_save()

def main():
    if args.mode=='train':
        args.round=0
        model = resnet.ResNet18(num_classes=10)
        train_model(model, epochs=100, lr=0.1, weights_only=False)
    elif args.mode=='prune':
        previous_ckpt = 'round-%d-best.pth'%(args.round-1)
        print("Pruning round %d, load model from %s"%( args.round, previous_ckpt ))
        model = torch.load( previous_ckpt )
        model = prune_model(model)
        print(model)
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters: %.1fM`"%(params/1e6))
        train_model(model, epochs=30, lr=0.01, weights_only=False)
    elif args.mode=='test':
        train_loader, test_loader = get_dataloader()
        evaluator = engine.evaluator.ClassificationEvaluator(test_loader)
        ckpt = 'round-%d-best.pth'%(args.round)
        print("Load model from %s"%( ckpt ))
        model = torch.load( ckpt )
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters: %.1fM"%(params/1e6))
        print("Acc=%.4f\n"%( evaluator.eval( model )['acc'] ) )

if __name__=='__main__':
    main()

