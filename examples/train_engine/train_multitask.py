import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn

from PIL import Image

from kamal import engine, metrics, loss

from kamal.vision.models.segmentation import segnet_vgg11_bn, segnet_vgg16_bn, segnet_vgg19_bn
from kamal.amalgamation.sbm import JointNet
from kamal.vision.datasets import NYUv2
from kamal.vision.datasets import LabelConcatDataset
from kamal.utils import Denormalizer
from kamal.vision import sync_transforms as sT


from visdom import Visdom
import random

# 1. Download dataset from https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid
# 2. Run visdom server: $ visdom -p 29999
# 3. python train_camvid.py --lr 0.01 --data_root /PATH/TO/CamVid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--data_root', type=str,
                        default='~/Datasets/NYUv2')
    parser.add_argument('--total_iters', type=int, default=20000)
    parser.add_argument('--seg_path', type=str,
                        default='~/checkpoints/segnet_vgg11_bn-best-00009200-mIoU-0.446.pth')
    parser.add_argument('--depth_path', type=str,
                        default='~/checkpoints/depth_segnet_vgg11_bn-latest-00019600-rmse-2.885.pth')
    args = parser.parse_args()

    # tasks
    tasks = ['Segmentation', 'Depth']
    # split_size
    split_size = [13, 1]  # num_classes

    # Prepare data
    train_dst_list = []
    test_dst_list = []
    for task_name in tasks:
        if task_name == 'Segmentation':
            train_dst = NYUv2(root=args.data_root,
                              split='train', target_type='semantic')
            test_dst = NYUv2(root=args.data_root, split='test',
                             target_type='semantic')
        if task_name == 'Depth':
            train_dst = NYUv2(root=args.data_root,
                              split='train', target_type='depth')
            test_dst = NYUv2(root=args.data_root, split='test',
                             target_type='depth')
        train_dst_list.append(train_dst)
        test_dst_list.append(test_dst)
    train_concat_dst = LabelConcatDataset(train_dst_list, tasks, transforms=sT.Compose([
        sT.Sync(sT.RandomCrop(320)),
        sT.Sync(sT.RandomRotation(5)),
        sT.Sync(sT.RandomHorizontalFlip()),
        sT.Multi(sT.ColorJitter(0.4, 0.4, 0.4), None, None),
        sT.Multi(sT.ToTensor(), sT.ToTensor(
            normalize=False, dtype=torch.long), sT.ToTensor(
            normalize=False, dtype=torch.float)),
        sT.Multi(sT.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225]), None, sT.Lambda(lambda depth: depth.float() / 1000.))
    ]))
    test_concat_dst = LabelConcatDataset(test_dst_list, tasks, transforms=sT.Compose([
        sT.Multi(sT.ToTensor(), sT.ToTensor(
            normalize=False, dtype=torch.long), sT.ToTensor(
            normalize=False, dtype=torch.float)),
        sT.Multi(sT.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225]), None, sT.Lambda(lambda depth: depth.float() / 1000.))
    ]))
    train_loader = torch.utils.data.DataLoader(
        train_concat_dst, batch_size=16, num_workers=2, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        test_concat_dst, batch_size=16, num_workers=2, pin_memory=True, shuffle=True)
    print('train: %s, val: %s' % (len(train_loader), len(val_loader)))

    # Prepare model
    teacher_seg_model = segnet_vgg11_bn(
        num_classes=13, pretrained_backbone=False)
    teacher_dep_model = segnet_vgg11_bn(
        num_classes=1, pretrained_backbone=False)
    joint_model = JointNet(
        [teacher_seg_model, teacher_dep_model],
        indices=[3, 1],
        phase='block'
    )
    teacher_seg_model.load_state_dict(torch.load(args.seg_path))
    teacher_dep_model.load_state_dict(torch.load(args.depth_path))
    task = engine.task.MultitaskTask(weights=[1, 1], criterions=[nn.CrossEntropyLoss(
        ignore_index=255), nn.L1Loss()], tasks=tasks)

    # Prepare trainer
    params_1x = []
    params_10x = []
    for name, param in joint_model.named_parameters():
        if 'fc' in name:
            params_10x.append(param)
        else:
            params_1x.append(param)

    optimizer = torch.optim.SGD(params=[{'params': params_1x,  'lr': args.lr},
                                        {'params': params_10x, 'lr': args.lr*10}],
                                lr=args.lr, weight_decay=1e-4, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.total_iters)
    evaluator = engine.evaluator.MultitaskEvaluator(
        val_loader, split_size=split_size, task=task, tasks=tasks)
    trainer = engine.trainer.MultitaskTrainer(
        task=task, model=joint_model, teachers=[teacher_seg_model, teacher_dep_model], split_size=split_size, viz=Visdom(port='19999', env='multitask'))

    trainer.add_callbacks([
        engine.callbacks.LoggingCallback(
            interval=10,
            names=('total_loss', 'lr'),
            smooth_window_sizes=(20, None)),
        engine.callbacks.LRSchedulerCallback(scheduler=scheduler),
        engine.callbacks.MultitaskValidationCallback(
            interval=400,
            split_size=split_size,
            evaluator=evaluator,
            save_model=('best', 'latest'),
            ckpt_dir='checkpoints',
            ckpt_tag='multitask_segnet_vgg11_bn_b'
        ),
        engine.callbacks.VisualizeMultitaskCallBack(
            interval=400,
            dataset=val_loader.dataset,
            split_size=split_size,
            tasks=tasks,
            idx_list_or_num_vis=5,  # select 5 images for visualization
            denormalizer=Denormalizer(mean=[0.485, 0.456, 0.406], std=[
                                      0.229, 0.224, 0.225]),
            scale_to_255=True)     # 0~1 => 0~255
    ])
    trainer.train(0, args.total_iters,
                  train_loader=train_loader, optimizer=optimizer)


if __name__ == '__main__':
    main()
