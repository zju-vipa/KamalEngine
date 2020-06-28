import argparse
import torch
import torch.nn as nn

from kamal import engine, metrics, vision, amalgamation, utils
from kamal.vision import sync_transforms as sT

from visdom import Visdom


# 1. Download dataset from https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid
# 2. Run visdom server: $ visdom -p 29999
# 3. python train_camvid.py --lr 0.01 --data_root /PATH/TO/CamVid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--data_root', type=str,
                        default='/home/zzc/Datasets/NYUv2')
    parser.add_argument('--total_iters', type=int, default=40000)
    args = parser.parse_args()

    # Prepare data
    train_dst = vision.datasets.NYUv2(root=args.data_root, split='train', target_type='depth',
                      transforms=sT.Compose([
                          sT.Sync(sT.RandomRotation(5), sT.RandomRotation(5)),
                          sT.Sync(sT.RandomHorizontalFlip(),
                                  sT.RandomHorizontalFlip()),
                          sT.Multi(sT.ColorJitter(0.4, 0.4, 0.4), None),
                          sT.Multi(sT.ToTensor(), sT.ToTensor(
                              normalize=False, dtype=torch.float)),
                          sT.Multi(sT.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                   0.229, 0.224, 0.225]), sT.Lambda(lambda depth: depth.float() / 1000.))
                      ]),
                      )
    test_dst = vision.datasets.NYUv2(root=args.data_root, split='test', target_type='depth',
                     transforms=sT.Compose([
                         sT.Multi(sT.ToTensor(), sT.ToTensor(
                             normalize=False, dtype=torch.float)),
                         sT.Multi(sT.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                  0.229, 0.224, 0.225]), sT.Lambda(lambda depth: depth.float() / 1000.))
                     ]),
                     )
    train_loader = torch.utils.data.DataLoader(
        train_dst, batch_size=8, num_workers=2, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        test_dst, batch_size=8, num_workers=2, pin_memory=True, shuffle=True)
    print('train: %s, val: %s' % (len(train_loader), len(val_loader)))

    # Prepare model
    model = vision.models.segnet_vgg11_bn(num_classes=1, pretrained_backbone=True)
    task = engine.task.DepthTask(criterion=nn.L1Loss())

    # Prepare trainer
    params_1x = []
    params_10x = []
    for name, param in model.named_parameters():
        if 'fc' in name:
            params_10x.append(param)
        else:
            params_1x.append(param)

    optimizer = torch.optim.SGD(params=[{'params': params_1x,  'lr': args.lr},
                                        {'params': params_10x, 'lr': args.lr*10}],
                                lr=args.lr, weight_decay=1e-4, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.total_iters)
    evaluator = engine.evaluator.DepthEvaluator(val_loader)
    trainer = engine.trainer.SimpleTrainer(
        task=task, model=model, viz=Visdom(port='19999', env='depth_NYUv2_vgg11_bn'))

    trainer.add_callbacks([
        engine.callbacks.LoggingCallback(
            interval=10,
            names=('total_loss', 'lr'),
            smooth_window_sizes=(20, None)),
        engine.callbacks.LRSchedulerCallback(scheduler=scheduler),
        engine.callbacks.ValidationCallback(
            interval=800,
            evaluator=evaluator,
            save_model=('best', 'latest'),
            ckpt_dir='checkpoints',
            ckpt_tag='depth_segnet_vgg19_bn'
        ),
        engine.callbacks.VisualizeDepthCallBack(
            interval=800,
            dataset=val_loader.dataset,
            idx_list_or_num_vis=5,  # select 5 images for visualization
            denormalizer=utils.Denormalizer(mean=[0.485, 0.456, 0.406], std=[
                                      0.229, 0.224, 0.225]),
            scale_to_255=True)     # 0~1 => 0~255
    ])
    trainer.train(0, args.total_iters,
                  train_loader=train_loader, optimizer=optimizer)


if __name__ == '__main__':
    main()
