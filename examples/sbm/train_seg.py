import argparse
import os, sys
import torch
import torch.nn as nn

from PIL import Image

from kamal import engine, metrics

from kamal.vision.models.segmentation import segnet_vgg11_bn
from kamal.vision.datasets import NYUv2
from kamal.vision import sync_transforms as sT
from kamal.utils import Denormalizer

from visdom import Visdom
import random

# 1. Download dataset from https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid
# 2. Run visdom server: $ visdom -p 29999
# 3. python train_camvid.py --lr 0.01 --data_root /PATH/TO/CamVid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--data_root', type=str, default='./data/NYUv2')
    parser.add_argument('--total_iters', type=int, default=10000)
    args = parser.parse_args()

    # Prepare data
    train_loader = torch.utils.data.DataLoader(
        NYUv2(args.data_root, split='train', transforms=sT.Compose([
            sT.Sync(  sT.RandomCrop(320) ),
            sT.Sync(  sT.RandomHorizontalFlip() ),
            sT.Multi( sT.ColorJitter(0.4, 0.4, 0.4), None ),
            sT.Multi( sT.ToTensor(normalize=True), sT.ToTensor(normalize=False, dtype=torch.long) ),
            sT.Multi( sT.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None )
        ] )), batch_size=16, num_workers=2, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        NYUv2(args.data_root, split='test', transforms=sT.Compose([
            sT.Multi(sT.ToTensor(normalize=True),sT.ToTensor(normalize=False, dtype=torch.long) ),
            sT.Multi(sT.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None )
        ] )), batch_size=16, num_workers=2)

    # Prepare model
    model = segnet_vgg11_bn(num_classes=13, pretrained_backbone=True)
    task = engine.task.SegmentationTask( criterion=nn.CrossEntropyLoss(ignore_index=255) )

    # Prepare trainer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.total_iters)
    evaluator = engine.evaluator.SegmentationEvaluator( 13, val_loader )
    trainer = engine.trainer.SimpleTrainer( task=task, model=model, viz=Visdom(port='29999', env='test') )
    
    trainer.add_callbacks( [
        engine.callbacks.LoggingCallback(
            interval=10,
            names=('total_loss', 'lr' ), 
            smooth_window_sizes=( 20, None )),
        engine.callbacks.LRSchedulerCallback(scheduler=scheduler),
        engine.callbacks.ValidationCallback(
            interval=1, 
            evaluator=evaluator,
            save_model=('best', 'latest'), 
            reverse = True,
            ckpt_dir='checkpoints',
            ckpt_tag='segnet_vgg11_bn'
        ),
        engine.callbacks.VisualizeSegmentationCallBack(
            interval=1,
            dataset=val_loader.dataset,
            idx_list_or_num_vis=5, # select 5 images for visualization
            denormalizer=Denormalizer(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            scale_to_255=True)     # 0~1 => 0~255
    ] )
    trainer.train(0, args.total_iters, train_loader=train_loader, optimizer=optimizer)

if __name__=='__main__':
    main()