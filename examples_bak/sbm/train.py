import argparse
import torch
import torch.nn as nn

from kamal import engine, metrics, vision, amalgamation, utils
from kamal.vision import sync_transforms as sT

from visdom import Visdom
import random



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--data_root', type=str,
                        default='~/Datasets/NYUv2')
    parser.add_argument('--total_iters', type=int, default=40000)
    parser.add_argument('--seg_path', type=str,
                        default='~/segnet_seg.pth')
    parser.add_argument('--depth_path', type=str,
                        default='~/segnet_depth.pth')
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
            train_dst = vision.datasets.NYUv2(root=args.data_root,
                              split='train', target_type='semantic')
            test_dst = vision.datasets.NYUv2(root=args.data_root, split='test',
                             target_type='semantic')
        if task_name == 'Depth':
            train_dst = vision.datasets.NYUv2(root=args.data_root,
                              split='train', target_type='depth')
            test_dst = vision.datasets.NYUv2(root=args.data_root, split='test',
                             target_type='depth')
        train_dst_list.append(train_dst)
        test_dst_list.append(test_dst)
    train_concat_dst = vision.datasets.LabelConcatDataset(train_dst_list, tasks, transforms=sT.Compose([
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
    test_concat_dst = vision.datasets.LabelConcatDataset(test_dst_list, tasks, transforms=sT.Compose([
        sT.Multi(sT.ToTensor(), sT.ToTensor(
            normalize=False, dtype=torch.long), sT.ToTensor(
            normalize=False, dtype=torch.float)),
        sT.Multi(sT.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225]), None, sT.Lambda(lambda depth: depth.float() / 1000.))
    ]))
    train_loader = torch.utils.data.DataLoader(
        train_concat_dst, batch_size=8, num_workers=2, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        test_concat_dst, batch_size=8, num_workers=2, pin_memory=True, shuffle=True)
    print('train: %s, val: %s' % (len(train_loader), len(val_loader)))

    # Prepare model
    teacher_seg_model = vision.models.segmentation.segnet_vgg11_bn(
        num_classes=13, pretrained_backbone=False)
    teacher_seg_model.load_state_dict(torch.load(args.seg_path))
    teacher_dep_model = vision.models.segmentation.segnet_vgg11_bn(
        num_classes=1, pretrained_backbone=False)
    teacher_dep_model.load_state_dict(torch.load(args.depth_path))
    joint_model = amalgamation.sbm.JointNet(
        [teacher_seg_model, teacher_dep_model],
        indices=[3, 1],
        phase='block'
    )
    task = engine.task.SbmTask(criterions=[nn.CrossEntropyLoss(
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
    evaluator = engine.evaluator.SbmEvaluator(
        val_loader, split_size=split_size, task=task, tasks=tasks)
    trainer = amalgamation.sbm.SbmTrainer(
        task=task, model=joint_model, teachers=[teacher_seg_model, teacher_dep_model], split_size=split_size, viz=Visdom(port='19999', env='sbm'))

    trainer.add_callbacks([
        engine.callbacks.LoggingCallback(
            interval=10,
            names=('total_loss', 'lr'),
            smooth_window_sizes=(20, None)),
        engine.callbacks.LRSchedulerCallback(scheduler=scheduler),
        engine.callbacks.SbmValidationCallback(
            interval=400,
            split_size=split_size,
            evaluator=evaluator,
            save_model=('best', 'latest'),
            ckpt_dir='checkpoints',
            ckpt_tag='sbm_segnet_vgg11_bn'
        ),
        engine.callbacks.VisualizeSbmCallBack(
            interval=400,
            dataset=val_loader.dataset,
            split_size=split_size,
            tasks=tasks,
            idx_list_or_num_vis=5,  # select 5 images for visualization
            denormalizer=utils.Denormalizer(mean=[0.485, 0.456, 0.406], std=[
                                      0.229, 0.224, 0.225]),
            scale_to_255=True)     # 0~1 => 0~255
    ])
    trainer.train(0, args.total_iters,
                  train_loader=train_loader, optimizer=optimizer)


if __name__ == '__main__':
    main()
