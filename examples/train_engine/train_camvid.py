import argparse
import os, sys
import torch
import torch.nn as nn

from PIL import Image

from kamal import engine, metrics
from kamal.vision.models.segmentation import deeplabv3plus_mobilenetv2
from kamal.vision.datasets import CamVid
from kamal.vision import sync_transforms as T
from visdom import Visdom
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--data_root', type=str, default='./data/camvid')
    args = parser.parse_args()

    train_loader = torch.utils.data.DataLoader(
        CamVid(args.data_root, split='trainval', transforms=T.Compose([
            T.Sync( T.RandomCrop(320) ),
            T.Sync( T.RandomHorizontalFlip() ),
            T.Multi( T.ColorJitter(0.5, 0.5, 0.5), None ),
            T.Multi(T.ToTensor(normalize=True), T.ToTensor(normalize=False, dtype=torch.long) ),
            T.Multi(T.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None )
        ] )), batch_size=8, num_workers=2, pin_memory=True, shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        CamVid(args.data_root, split='test', transforms=T.Compose([
            T.Multi(T.ToTensor(normalize=True),T.ToTensor(normalize=False, dtype=torch.long) ),
            T.Multi(T.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None )
        ] )), batch_size=8, num_workers=2)

    model = deeplabv3plus_mobilenetv2(num_classes=11, pretrained_backbone=True)
    #model = SegNet( arch='vgg11_bn', num_classes=11, pretrained_backbone=True )
    #model = LinkNet( arch='resnet34', num_classes=11, pretrained_backbone=True )
    task = engine.task.SegmentationTask( criterion=nn.CrossEntropyLoss(ignore_index=255) )
    #model = UNet(num_classes=11)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5000, 8000], gamma=0.1)
    trainer = engine.trainer.SimpleTrainer( task, model, train_loader, optimizer )

    NUM_VIS = 3
    sub_dataloader_for_vis = torch.utils.data.DataLoader( 
                        torch.utils.data.Subset( 
                            val_loader.dataset, 
                            indices=random.sample( list(range( len(val_loader.dataset) )), NUM_VIS),
                        ), batch_size=1, num_workers=2)
    
    viz = Visdom(port='29999', env='camvid')
    trainer.add_callbacks( [
        engine.callbacks.LoggingCallback(
            interval=10, 
            viz=viz, 
            names=('total_loss', ), 
            smooth_window_size=10),
        engine.callbacks.LRSchedulerCallback(scheduler),
        engine.callbacks.SimpleValidationCallback(
            interval=200, 
            val_loader=val_loader, 
            metrics=metrics.StreamSegmentationMetrics(11),
            save_model=('best', 'latest'), 
            ckpt_dir='checkpoints',
            ckpt_tag='deeplabv3plus_mobilenet_camvid',
            viz = viz,
        ),
        engine.callbacks.SegVisualizationCallback(
            interval=200, 
            viz=viz,
            data_loader=sub_dataloader_for_vis, 
            to_255=True,
            norm_mean=[0.485, 0.456, 0.406], 
            norm_std=[0.229, 0.224, 0.225])
    ] )
    trainer.train(0, 10000)

if __name__=='__main__':
    main()