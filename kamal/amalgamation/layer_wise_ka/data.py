# coding:utf-8
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch


def get_dataset(path, batchsize=8):

    train_set = datasets.ImageFolder(path,
                                     transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]))
    train_loader = DataLoader(
        train_set, batch_size=batchsize, shuffle=False, num_workers=4)
    return train_loader, len(train_set)
