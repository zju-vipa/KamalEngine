from kamal import vision, engine, utils, amalgamation, metrics, callbacks
from kamal.vision import sync_transforms as sT

import torch, time
from torch.utils.tensorboard import SummaryWriter

import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--lr', type=float, default=1e-1)
parser.add_argument( '--momentum', type=float, default=0.9)
parser.add_argument( '--T', type=int, default=4)
parser.add_argument( '--T_dis', type=int, default=4)
parser.add_argument( '--key_ramda', type=float, default=0.5)
parser.add_argument( '--ckpt', type=str, default=None)
args = parser.parse_args()

def main():
    train_dst = vision.datasets.torchvision_datasets.CIFAR100(
        'data/torchdata', train=True, download=True, transform=sT.Compose([
            sT.RandomCrop(32, padding=4),
            sT.RandomHorizontalFlip(),
            sT.ToTensor(),
            sT.Normalize( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) )
        ]) )
    val_dst = vision.datasets.torchvision_datasets.CIFAR100(
        'data/torchdata', train=False, download=True, transform=sT.Compose([
            sT.ToTensor(),
            sT.Normalize( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) )
        ]) )
    train_loader = torch.utils.data.DataLoader( train_dst, batch_size=128, shuffle=True, num_workers=4 )
    val_loader = torch.utils.data.DataLoader( val_dst, batch_size=128, num_workers=4 )

    Teacher = vision.models.classification.resnet18(num_classes=10, pretrained=False)
    Teacher_w_key = vision.models.classification.resnet18(num_classes=10, pretrained=False)
    student = vision.models.classification.resnet18(num_classes=10, pretrained=False)

    TOTAL_ITERS = len(train_loader) * 200  # default_epoch = 200
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    optim = torch.optim.SGD(student.parameters(), lr=args.lr, momentum=args.momentum)
