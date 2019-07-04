import random
from visualizer import Visualizer
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import os
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.realpath(__file__)))))

from kamal.losses import SoftCELoss
from kamal.metrics import MetrcisCompose
from kamal.common_feature import CommonFeatureLearning, CFL_ConvBlock
from kamal.core import AmalNet, LayerParser
from kamal.metrics import StreamClsMetrics
from kamal.datasets import StanfordDogs, CUB200
from kamal.models import resnet18, resnet34
import torch
import torch.nn as nn




def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='./data')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--random_seed", type=int, default=1337)
    parser.add_argument("--download", action='store_true', default=False)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--vis_port", type=str, default='13579')

    return parser


def kd(cur_epoch, criterion_ce, model, teachers, optim, train_loader, device, scheduler=None, print_interval=10, vis=None, trace_name=None):
    """Train and return epoch loss"""
    ta, tb = teachers

    if scheduler is not None:
        scheduler.step()

    print("Epoch %d, lr = %f" % (cur_epoch, optim.param_groups[0]['lr']))
    epoch_loss = 0.0
    interval_loss = 0.0

    for cur_step, (images, labels) in enumerate(train_loader):

        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)

        # N, C, H, W
        optim.zero_grad()
        with torch.no_grad():
            a_out = ta(images)
            b_out = tb(images)
            t_outs = torch.cat((a_out, b_out), dim=1)
        
        s_outs = model(images)

        loss = criterion_ce(s_outs, t_outs, None)

        loss.backward()
        optim.step()

        np_loss = loss.detach().cpu().numpy()

        epoch_loss += np_loss
        interval_loss += np_loss

        if (cur_step+1) % print_interval == 0:
            interval_loss = interval_loss/print_interval

            if vis is not None:
                x = cur_epoch*len(train_loader)+cur_step
                vis.vis_scalar('CE Loss', trace_name, x, interval_loss)

            print("Epoch %d, Batch %d/%d, Loss=%f" %
                  (cur_epoch, cur_step+1, len(train_loader), interval_loss))

    return epoch_loss / len(train_loader)


def validate(model, loader, device, metrics):
    """Do validation and return specified samples"""
    metrics.reset()
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)

            preds = outputs.detach()  # .max(dim=1)[1].cpu().numpy()
            targets = labels  # .cpu().numpy()

            metrics.update(preds, targets)
        score = metrics.get_results()
    return score


def main():
    opts = get_parser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vis = Visualizer(port=opts.vis_port, env='kd')

    # Set up random seed
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    cur_epoch = 0
    best_score = 0.0
    ckpt_dir = './checkpoints'
    latest_ckpt = 'checkpoints/kd_resnet34_latest.pth'
    best_ckpt = 'checkpoints/kd_resnet34_best.pth'

    #  Set up dataloader
    transforms_train = transforms.Compose([
        transforms.Resize(size=224),
        transforms.RandomCrop(size=(224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    transforms_val = transforms.Compose([
        transforms.Resize(size=224),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    
    cub_root = os.path.join(opts.data_root, 'cub200')
    train_cub = CUB200(root=cub_root, split='train',
                       transforms=transforms_train,
                       download=opts.download, offset=0)
    val_cub = CUB200(root=cub_root, split='test',
                     transforms=transforms_val,
                     download=False, offset=0)

    dogs_root = os.path.join(opts.data_root, 'dogs')
    train_dogs = StanfordDogs(root=dogs_root, split='train',
                              transforms=transforms_train,
                              download=opts.download, offset=200)
    val_dogs = StanfordDogs(root=dogs_root, split='test',
                            transforms=transforms_val,
                            download=False, offset=200) # add offset

    train_dst = data.ConcatDataset([train_cub, train_dogs])
    val_dst = data.ConcatDataset([val_cub, val_dogs])

    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, drop_last=True, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.batch_size, drop_last=True, shuffle=False, num_workers=4)

    # pretrained teachers
    print("Loading pretrained teachers ...")
    cub_teacher_ckpt = 'checkpoints/cub200_resnet18_best.pth'
    dogs_teacher_ckpt = 'checkpoints/dogs_resnet34_best.pth'
    t_cub = resnet18(num_classes=200).to(device)
    t_dogs = resnet34(num_classes=120).to(device)
    t_cub.load_state_dict(torch.load(cub_teacher_ckpt)['model_state'])
    t_dogs.load_state_dict(torch.load(dogs_teacher_ckpt)['model_state'])
    t_cub.eval()
    t_dogs.eval()

    num_classes = 120+200
    stu = resnet34(pretrained=True, num_classes=num_classes).to(device)
    metrics = StreamClsMetrics(num_classes)

    params_1x = []
    params_10x = []
    for name, param in stu.named_parameters():
        if 'fc' in name:
            params_10x.append(param)
        else:
            params_1x.append(param)
    optimizer = torch.optim.Adam([{'params': params_1x,         'lr': opts.lr},
                                  {'params': params_10x,        'lr': opts.lr*10}, ],
                                 lr=opts.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=30, gamma=0.1)
    
    criterion_ce = SoftCELoss(T=1.0)
    def save_ckpt(path):
        """ save current model
        """
        state = {
            "epoch": cur_epoch,
            "model_state": stu.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }
        torch.save(state, path)
        print("Model saved as %s" % path)

    print("Training ...")
    # ===== Train Loop =====#
    while cur_epoch < opts.epochs:
        stu.train()
        epoch_loss = kd(cur_epoch=cur_epoch,
                        criterion_ce=criterion_ce,
                        model=stu,
                        teachers=[t_cub, t_dogs],
                        optim=optimizer,
                        train_loader=train_loader,
                        device=device,
                        scheduler=scheduler,
                        vis=vis,
                        trace_name='kd')

        print("End of Epoch %d/%d, Average Loss=%f" %
              (cur_epoch, opts.epochs, epoch_loss))

        # =====  Latest Checkpoints  =====
        save_ckpt(latest_ckpt)
#
        # =====  Validation  =====
        print("validate on val set...")
        stu.eval()
        val_score = validate(model=stu,
                             loader=val_loader,
                             device=device,
                             metrics=metrics)
        print(metrics.to_str(val_score))
#
        # =====  Save Best Model  =====
        if val_score['Overall Acc'] > best_score:  # save best model
            best_score = val_score['Overall Acc']
            save_ckpt(best_ckpt)
#
        vis.vis_scalar('Overall Acc', 'kd', cur_epoch +
                       1, val_score['Overall Acc'])
        cur_epoch += 1

if __name__ == '__main__':
    main()
