import os
import pickle
import sys
import random

import numpy as np
from PIL import Image

from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms


def get_normalizer(data_set, inverse=False):
    if data_set == 'CIFAR10':
        MEAN = (0.4914, 0.4822, 0.4465)
        STD = (0.2023, 0.1994, 0.2010)

    elif data_set == 'CIFAR100':
        MEAN = (0.5071, 0.4867, 0.4408)
        STD = (0.2675, 0.2565, 0.2761)

    elif data_set == 'CIFAR10Few':
        MEAN = (0.4914, 0.4822, 0.4465)
        STD = (0.2023, 0.1994, 0.2010)
    elif data_set == 'CIFAR10Few':
        MEAN = (0.5071, 0.4867, 0.4408)
        STD = (0.2675, 0.2565, 0.2761)
    else:
        raise RuntimeError("Not expected data flag !!!")

    if inverse:
        MEAN = [-mean / std for mean, std in zip(MEAN, STD)]
        STD = [1 / std for std in STD]

    return transforms.Normalize(MEAN, STD)


def get_transformer(data_set, imsize=None, cropsize=None,
                    crop_padding=None, hflip=None):
    transformers = []
    if imsize:
        transformers.append(transforms.Resize(imsize))
    if cropsize:
        transformers.append(
            transforms.RandomCrop(cropsize, padding=crop_padding))
    if hflip:
        transformers.append(transforms.RandomHorizontalFlip())

    transformers.append(transforms.ToTensor())
    transformers.append(get_normalizer(data_set))

    return transforms.Compose(transformers)


def get_dataset(args, train_flag=True):
    if train_flag:
        dataset = torchvision.datasets.__dict__[args.dataset] \
            (root=args.data_path, train=True,
             transform=get_transformer(args.dataset, args.imsize,
                                       args.cropsize,
                                       args.crop_padding,
                                       args.hflip), download=True)
    else:
        dataset = torchvision.datasets.__dict__[args.dataset] \
            (root=args.data_path, train=False,
             transform=get_transformer(args.dataset), download=True)
    return dataset


class CIFARFew(Dataset):
    def __init__(self, root,entry,transform=None):
        self.root = root
        self.transform = transform
        self.data = entry
        self.data = self.data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.data)






