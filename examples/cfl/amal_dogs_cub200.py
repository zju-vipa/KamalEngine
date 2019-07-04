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
    parser.add_argument("--vis_port", type=str, default='13570')

    return parser

def main():
    opts = get_parser().parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up random seed

    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    ckpt_dir = './checkpoints'
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

    train_dst = data.ConcatDataset([train_cub, train_dogs])
    val_dst = data.ConcatDataset([val_cub, val_dogs])

    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, drop_last=True,shuffle=True, num_workers=4)
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.batch_size, drop_last=True,shuffle=False, num_workers=4)

    print("Loading pretrained teachers ...")
    cub_teacher_ckpt = 'checkpoints/cub200_resnet18_best.pth'
    dogs_teacher_ckpt = 'checkpoints/dogs_resnet34_best.pth'
    t_cub = resnet18(num_classes=200)
    t_dogs = resnet34(num_classes=120)
    t_cub.load_state_dict(torch.load(cub_teacher_ckpt)['model_state'])
    t_dogs.load_state_dict(torch.load(dogs_teacher_ckpt)['model_state'])

    num_classes = 120+200
    stu = resnet34(pretrained=True, num_classes=num_classes).to(device)
    metrics = StreamClsMetrics(num_classes)

    t_cub = AmalNet(t_cub)
    t_dogs = AmalNet(t_dogs)
    stu = AmalNet(stu)

    def resnet_parse_fn(resnet):
        for l in [resnet.layer4]: #[resnet.layer3, resnet.layer4]:
            yield l, l[-1].bn2.num_features
    
    t_cub.register_endpoints(parse_fn=resnet_parse_fn)
    t_dogs.register_endpoints(parse_fn=resnet_parse_fn)
    stu.register_endpoints(parse_fn=resnet_parse_fn)
    teacher = [t_cub, t_dogs]

    def get_cfl_block(student_model, teacher_model, channel_h=128):
        t_channels = zip(*[t.endpoints_info for t in teacher_model])
        s_channels = student_model.endpoints_info

        cfl_blocks = list()
        for s_ch, t_ch in zip(s_channels, t_channels):
            cfl_blocks.append(CFL_ConvBlock(
                channel_s=s_ch, channel_t=t_ch, channel_h=channel_h))
        return cfl_blocks

    cfl = CommonFeatureLearning(
        stu, teacher, get_cfl_block(stu, teacher, channel_h=128)).to(device)

    params_1x = []
    params_10x = []
    for name, param in cfl.student.named_parameters():
        if 'fc' in name:
            params_10x.append(param)
        else:
            params_1x.append(param)
    params_cfl = cfl.cfl_blocks.parameters()
    optimizer = torch.optim.Adam([{'params': params_1x,         'lr': opts.lr},
                                  {'params': params_10x,        'lr': opts.lr*10},
                                  {'params': params_cfl,        'lr':  opts.lr*10}],
                                 lr=opts.lr, weight_decay=1e-4, betas=(0.8, 0.9)) 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    print("CFL Training ...")
    cfl.fit(train_loader,
            val_loader=val_loader,
            metrics=MetrcisCompose([metrics]),
            beta=10.0,
            total_epochs=opts.epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            enable_vis=True,
            env='cfl',
            port='13579',
            gpu_id='0')

if __name__ == '__main__':
    main()
