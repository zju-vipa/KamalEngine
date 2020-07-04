import sys, os
import cifar_resnet as resnet

from kamal import slim
from kamal.slim import torch_pruning as tp
from kamal import engine, vision
from kamal.vision import sync_transforms as sT

import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn 
import numpy as np 

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True, choices=['train', 'prune', 'test'])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--total_epochs', type=int, default=100)
parser.add_argument('--step_size', type=int, default=70)
parser.add_argument('--round', type=int, default=1)

args = parser.parse_args()

def get_dataloader():
    train_dst = vision.datasets.CamVid( 
        '../data/CamVid11', split='trainval', transforms=sT.Compose([
            sT.Multi( sT.Resize(240),     sT.Resize(240)),
            sT.Sync(  sT.RandomCrop(240), sT.RandomCrop(240) ),
            sT.Sync(  sT.RandomHorizontalFlip(), sT.RandomHorizontalFlip() ),
            sT.Multi( sT.ToTensor(), sT.ToTensor( normalize=False, dtype=torch.long ) ),
            sT.Multi( sT.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ), None )
        ]) )
    test_dst = vision.datasets.CamVid( 
        '../data/CamVid11', split='test', transforms=sT.Compose([
            sT.Multi( sT.Resize(240), sT.Resize(240)),
            sT.Multi( sT.ToTensor(),  sT.ToTensor( normalize=False, dtype=torch.long ) ),
            sT.Multi( sT.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ), None )
        ]) )
    train_loader = torch.utils.data.DataLoader(train_dst, batch_size=16, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dst, batch_size=16, shuffle=False, num_workers=4)
    return train_loader, test_loader

def prune_model(model, rate=0.2):
    prunner = slim.Pruner( slim.prunning.LNStrategy(n=1) )
    model = prunner.prune( model, rate=rate, example_inputs=torch.randn(1, 3, 240, 240) )
    return model    

def train_model(model, epochs=100, lr=0.1, weights_only=False):
    train_loader, test_loader = get_dataloader()
    trainer = engine.trainer.BasicTrainer()
    task = engine.task.SegmentationTask()

    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR( optim, step_size=epochs*len(train_loader)//3, gamma=0.1 )
    evaluator = engine.evaluator.SegmentationEvaluator(11, test_loader)
    trainer.setup(model, task, data_loader=train_loader, optimizer=optim)
    trainer.add_callbacks([
        engine.callbacks.LRSchedulerCallback( scheduler=[sched] ),
        engine.callbacks.ValidationCallback(
            interval=len(train_loader), 
            evaluator=evaluator, 
            weights_only=weights_only,
            save_type=('best', ),
            ckpt_tag='camvid-round-%s'%args.round)
    ])
    trainer.run( 0, epochs*len(train_loader) )
    trainer.callbacks[1].final_save()

def main():
    if args.mode=='train':
        args.round=0
        model = vision.models.segmentation.deeplabv3_resnet50(num_classes=11, pretrained_backbone=True)
        train_model(model, epochs=100, lr=0.01, weights_only=False)
    elif args.mode=='prune':
        previous_ckpt = 'checkpoints/camvid-round-%d-best.pth'%(args.round-1)
        print("Pruning round %d, load model from %s"%( args.round, previous_ckpt ))
        model = torch.load( previous_ckpt )
        model = prune_model(model)
        print(model)
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters: %.1fM`"%(params/1e6))
        train_model(model, epochs=20, lr=0.01, weights_only=False)
    elif args.mode=='test':
        train_loader, test_loader = get_dataloader()
        evaluator = engine.evaluator.ClassificationEvaluator(test_loader)
        ckpt = 'checkpoints/camvid-round-%d-best.pth'%(args.round)
        print("Load model from %s"%( ckpt ))
        model = torch.load( ckpt )
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters: %.1fM"%(params/1e6))
        print("Acc=%.4f\n"%( evaluator.eval( model )['acc'] ) )

if __name__=='__main__':
    main()

