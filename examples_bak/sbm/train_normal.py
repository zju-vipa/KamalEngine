from __future__ import division
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from kamal.vision.models import SegNet
from kamal.vision.datasets import NYUv2
from kamal.criterion import AngleLoss
from kamal.metrics import NormalPredictionMetrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import visdom

import argparse
import os

import numpy as np
from tqdm import tqdm
import random 
from torchvision import transforms
from torch.utils import data
import torchvision.models as models


vis = visdom.Visdom(env='normal')

def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def train( cur_epoch, criterion, model, optim, train_loader, device, scheduler=None, print_interval=10):
    """Train and return epoch loss"""
    
    if scheduler is not None:
        scheduler.step()
    
    print("Epoch %d, lr = %f"%(cur_epoch, optim.param_groups[0]['lr']))
    epoch_loss = 0.0
    interval_loss = 0.0

    for cur_step, sample_batched in enumerate( train_loader ):
        images = sample_batched[0].to(device, dtype=torch.float32)
        normals = sample_batched[3].to(device, dtype=torch.float32)
        masks = sample_batched[4].to(device, dtype=torch.float32)

        # N, C, H, W
        optim.zero_grad()
        outputs = model(images)

        pred_normals = F.normalize(outputs, dim=1)
        normals = F.normalize(normals, dim=1)
        loss = criterion(pred_normals, normals, masks=masks)
        
        loss.backward()
        optim.step()

        np_loss = loss.data.cpu().numpy()
        epoch_loss+=np_loss
        interval_loss+=np_loss
        pre_steps = cur_epoch * len(train_loader)

        if (cur_step+1)%print_interval==0:
            interval_loss = interval_loss/print_interval
            print("Epoch %d, Batch %d/%d, Loss=%f"%(cur_epoch, cur_step+1, len(train_loader), interval_loss))

            vis_images = sample_batched[0].numpy()
            vis_labels = (sample_batched[3].numpy() + 1) / 2.
            vis_masks = sample_batched[4].numpy()
            vis_preds = ((pred_normals + 1) / 2).data.cpu().numpy()

            vis.images(vis_images, nrow=3, win='images')
            vis.images(vis_labels, nrow=3, win='labels')
            vis.images(vis_masks, nrow=3, win='masks')
            vis.images(vis_preds, nrow=3, win='predictions')
            vis.line(X=[cur_step + pre_steps], Y=[interval_loss], win='interval_loss', update='append' if (cur_step + pre_steps) else None, opts=dict(title='interval_loss'))
            vis.line(X=[cur_step + pre_steps], Y=[optim.param_groups[0]['lr']], win='learning_rate', update='append' if (cur_step + pre_steps) else None, opts=dict(title='learning_rate'))

            interval_loss=0.0

    return epoch_loss / len(train_loader)

def validate( model, loader, device, metrics):
    """Do validation and return specified samples"""
    metrics.reset()
    with torch.no_grad():
        for i, (images, labels, depths, normals, masks) in  tqdm(enumerate(loader)):
            images = images.to(device, dtype=torch.float32)
            targets = normals.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            outputs = model(images)

            preds = F.normalize(outputs, dim=1)
            targets = F.normalize(targets, dim=1)

            preds = preds.data.cpu().numpy()
            targets = targets.data.cpu().numpy()
            masks = masks.data.cpu().numpy()

            metrics.update(preds, targets, masks)

    score = metrics.get_results()
    return score
    
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='~/Datasets/NYUv2')
    parser.add_argument("--dataset", type=str, default='nyu')
    parser.add_argument("--batch_size", type=int, default=4 )
    parser.add_argument("--lr", type=float, default=1e-2 )
    parser.add_argument("--step_size", type=float, default=70 )
    parser.add_argument("--gamma", type=float, default=0.3 )
    parser.add_argument("--gpu_id", type=str, default='1' )
    parser.add_argument("--random_seed", type=int, default=1337 )
    parser.add_argument("--download", action='store_true', default=False )
    parser.add_argument("--epochs", type=int, default=100 )
    parser.add_argument("--init_ckpt", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default='./checkpoints/normal/')

    return parser

def main():
    opts = get_parser().parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

    # Set up random seed
    set_seed(opts.random_seed)
    ckpt_dir = os.path.join(opts.ckpt, 'lr{}_stepsize{}_gamma{}'.format(opts.lr, opts.step_size, opts.gamma))
    mkdir(ckpt_dir)

    if opts.dataset == 'nyu':
        num_classes = 40
        train_ds = NYUv2(opts.data_root, 'train', num_classes,
                    transforms=transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.ColorJitter(0.7),
                        transforms.ToTensor()
                    ]),
                    target_transforms=[
                        transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.Lambda(lambda target: torch.from_numpy(np.array(target)))
                        ]),
                        transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.Lambda(lambda depth: torch.from_numpy(np.array(depth)).float() / 1000.)
                        ]),
                        transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Lambda(lambda normal: normal * 2 - 1)
                        ]),
                        transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                        ]),
                    ], ds_type='labeled')
        val_ds = NYUv2(opts.data_root, 'test', num_classes, 
                    transforms=transforms.Compose([
                        transforms.ToTensor()
                    ]),
                    target_transforms=[
                        transforms.Lambda(lambda target: torch.from_numpy(np.array(target))),
                        transforms.Lambda(lambda depth: torch.from_numpy(np.array(depth)).float() / 1000.),
                        transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Lambda(lambda normal: normal * 2 - 1)
                        ]),
                        transforms.ToTensor()
                    ], ds_type='labeled')
    else:
        pass

    train_loader = data.DataLoader(train_ds, batch_size=opts.batch_size, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(val_ds, batch_size=opts.batch_size, shuffle=False, num_workers=4)

    model = SegNet(n_classes=3, positive=False)
    vgg16 = models.vgg16(pretrained=True)
    model.init_vgg16_params(vgg16)
    model = model.to(device)

    metrics = NormalPredictionMetrics(thresholds=[11.25, 22.5, 30])

    params_1x = []
    params_10x = []
    for name, param in model.named_parameters():
        if 'fc' in name:
            params_10x.append(param)
        else:
            params_1x.append(param)

    optimizer = torch.optim.SGD(params=[{'params': params_1x,  'lr': opts.lr  },
                                        {'params': params_10x, 'lr': opts.lr*10 }],
                                   lr=opts.lr, weight_decay=1e-5, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=opts.gamma)
    criterion = AngleLoss()

    # Restore
    best_score = 0.0
    cur_epoch = 0
    if opts.init_ckpt is not None and os.path.isfile(opts.init_ckpt):
        checkpoint = torch.load(opts.init_ckpt, map_location='cpu')
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        cur_epoch = checkpoint["epoch"]+1
        print("Model restored from %s"%opts.init_ckpt)
        del checkpoint # free memory
    else:
        print("[!] No Restoration")

    def save_ckpt(path):
        """ save current model
        """
        state = {
                    "epoch": cur_epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
        }
        torch.save(state, path)
        print( "Model saved as %s"%path )

    while cur_epoch < opts.epochs:
        model.train()
        epoch_loss = train(cur_epoch=cur_epoch, 
                            criterion=criterion, 
                            model=model, 
                            optim=optimizer, 
                            train_loader=train_loader, 
                            device=device, 
                            scheduler=scheduler)
        
        print("End of Epoch %d/%d, Average Loss=%f"%(cur_epoch, opts.epochs, epoch_loss))
        
        save_ckpt(os.path.join(ckpt_dir, '{}.pth'.format(cur_epoch)))

        print("validate on val set...")
        model.eval()
        val_score = validate(model=model,
                                loader=val_loader, 
                                device=device, 
                                metrics=metrics)
        print(metrics.to_str(val_score))
        cur_epoch+=1
    
    
if __name__=='__main__':
    main()


