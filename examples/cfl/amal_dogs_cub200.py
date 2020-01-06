from tqdm import tqdm
import numpy as np
import argparse
import sys, os, random

import torch
import torch.nn as nn
import torch.nn.functional as F 

from torchvision import transforms
from torchvision.models import resnet18, resnet34

from kamal.amalgamation.common_feature import CommonFeatureLearning
from kamal.metrics import StreamClassificationMetrics
from kamal.vision.datasets import StanfordDogs, CUB200
from kamal.utils import VisdomPlotter
from kamal.loss import KDLoss
from kamal.core import train_tools

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
    return parser

def main():
    opts = get_parser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vp = VisdomPlotter(port='15550', env='cfl')

    # Set up random seed
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    ckpt_dir = 'checkpoints'
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
    
    # Set up dataloader
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
                            download=False, offset=200)
    train_dst = torch.utils.data.ConcatDataset([train_cub, train_dogs])
    val_dst = torch.utils.data.ConcatDataset([val_cub, val_dogs])
    train_loader = torch.utils.data.DataLoader(train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dst, batch_size=opts.batch_size, shuffle=False, num_workers=2)
    
    # Teachers and student
    print("Loading pretrained teachers ...")
    cub_teacher_ckpt = 'checkpoints/cub200_resnet18.pth'
    dogs_teacher_ckpt = 'checkpoints/dogs_resnet34.pth'
    t_cub = resnet18(num_classes=200)
    t_dogs = resnet34(num_classes=120)
    t_cub.load_state_dict(torch.load(cub_teacher_ckpt))
    t_dogs.load_state_dict(torch.load(dogs_teacher_ckpt))

    num_classes = 200+120
    stu = resnet34(pretrained=True)
    stu.fc = nn.Linear( stu.fc.in_features, num_classes )

    stu.train().to(device)
    t_cub.eval().to(device)
    t_dogs.eval().to(device)
    # CFL block
    cfl = CommonFeatureLearning( layers=(stu.layer4, t_cub.layer4, t_dogs.layer4), num_features=(512, 512, 512) ).train().to(device)

    metrics = StreamClassificationMetrics()

    params_1x = []
    params_10x = []
    for name, param in stu.named_parameters():
        if 'fc' in name:
            params_10x.append(param)
        else:
            params_1x.append(param)

    optimizer = torch.optim.Adam([{'params': params_1x,         'lr': opts.lr},
                                  {'params': params_10x,        'lr': opts.lr*10},
                                  {'params': cfl.parameters(),  'lr': opts.lr}], 
                                  lr=opts.lr, weight_decay=1e-4) 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    cur_epoch = 0
    best_score = 0.0
    print("CFL Training ...")
    kd_criterion = KDLoss()
    # ===== Train Loop =====#
    while cur_epoch < opts.epochs:
        stu.train()
        for i, (data, _) in enumerate( train_loader ):
            optimizer.zero_grad()
            data = data.to(device)
            t_cub_out = t_cub(data)
            t_dog_out = t_dogs(data)
            s_out = stu(data)

            kd_loss = kd_criterion( s_out, torch.cat( [t_cub_out, t_dog_out], dim=1 ).detach() )
            cfl_loss = cfl()
            loss = kd_loss + 20*cfl_loss
            
            loss.backward()
            optimizer.step()
            if i%10==0:
                print("Epoch %d/%d, Batch %d/%d Loss=%.4f (kd_loss=%.4f, cfl_loss=%.4f)" %(cur_epoch, opts.epochs, i, len(train_loader), loss.item(), kd_loss.item(), cfl_loss.item()))
            
        scheduler.step()
        # =====  Validation  =====
        print("validate on val set...")
        stu.eval()
        (metric_name, score), val_loss = train_tools.eval(model=stu,
                                                            criterion=nn.CrossEntropyLoss(),
                                                            test_loader=val_loader,
                                                            metric=metrics, device=device)
        print("%s: %.4f"%(metric_name, score))
        vp.add_scalar( 'acc', cur_epoch+1, score )

        # =====  Save Best Model  =====
        if score > best_score:  # save best model
            best_score = score
            torch.save( stu.state_dict(), 'checkpoints/cfl_resnet34.pth')
            torch.save( cfl.state_dict(), 'checkpoints/cfl_block.pth' )

        vp.add_scalar('acc', cur_epoch + 1, score)
        cur_epoch += 1

if __name__ == '__main__':
    main()
