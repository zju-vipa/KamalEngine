#coding:utf-8
from .utils import colormap, set_seed

import os
import torch
import torch.utils.data as data
from PIL import Image
from scipy.io import loadmat
import numpy as np
import glob
from torchvision import transforms
import random

NYU_DEPTH_BIN = 50
NYU_LENGTH_BIN = 0.14

class NYUv2(data.Dataset):
    """NYUv2 depth dataset loader.
    
    **Parameters:**
        - **root** (string): Root directory path.
        - **split** (string, optional): 'train' for training set, and 'test' for test set. Default: 'train'.
        - **num_classes** (string, optional): The number of classes, must be 40 or 13. Default:13.
        - **transform** (callable, optional): A function/transform that takes in an PIL image and returns a transformed version. Default: None.
        - **target_transforms** (callable, optional): A list of function/transform that takes in the target and transform it. Default: None.
        - **ds_type** (string, optional): To pick samples with labels or not. Default: 'labeled'.
    """
    cmap = colormap()

    def __init__(self,
                 root,
                 split='train',
                 num_classes=13,
                 transforms=None,
                 target_transforms=None,
                 ds_type='labeled'):

        assert(split in ('train', 'test'))
        assert(ds_type in ('labeled', 'unlabeled'))
        self.root = root
        self.split = split
        self.ds_type = ds_type
        self.transforms = transforms
        self.target_transforms = target_transforms

        self.num_classes = num_classes

        self.train_idx = np.array([255, ] + list(range(num_classes)))

        if ds_type == 'labeled':
            split_mat = loadmat(os.path.join(
                self.root, 'CODE', 'nyuv2-meta-data', 'splits.mat'))

            idxs = split_mat[self.split+'Ndxs'].reshape(-1)
            self.images = [os.path.join(self.root, '480_640', 'IMAGE', '%04d.png' % idx)
                           for idx in idxs]
            if self.num_classes == 13:
                self.targets = [os.path.join(self.root, 'CODE', 'nyuv2-meta-data', '%s_labels_13' % self.split, 'new_nyu_class13_%04d.png' % idx)
                                for idx in idxs]
            elif self.num_classes == 40:
                self.targets = [os.path.join(self.root, '480_640', 'SEGMENTATION', '%04d.png' % idx)
                                for idx in idxs]
            else:
                raise ValueError(
                    'Invalid number of classes! Please use 13 or 40')

            self.depths = [os.path.join(
                self.root, 'FINAL_480_640', 'DEPTH', '%04d.png' % idx) for idx in idxs]
            self.normals = [os.path.join(
                self.root, 'normals_gt', 'normals', '%04d.png' % (idx-1)) for idx in idxs]
            self.masks = [os.path.join(
                self.root, 'normals_gt', 'masks', '%04d.png' % (idx-1)) for idx in idxs]
        else:
            self.images = [glob.glob(os.path.join(
                self.root, 'unlabeled_images/*.png'))]

    def __getitem__(self, idx):
        if self.ds_type == 'labeled':
            image = Image.open(self.images[idx])
            target = Image.open(self.targets[idx])
            depth = Image.open(self.depths[idx])
            normal = Image.open(self.normals[idx])
            mask = Image.open(self.masks[idx])
            mask = mask.convert('L')

            seed = np.random.randint(2147483647)
            if self.transforms:
                set_seed(seed)
                image = self.transforms(image)
            
            if self.target_transforms:
                set_seed(seed)
                target = self.target_transforms[0](target)
                set_seed(seed)
                depth = self.target_transforms[1](depth)
                set_seed(seed)
                normal = self.target_transforms[2](normal)
                set_seed(seed)
                mask = self.target_transforms[3](mask)

            target = self.train_idx[target]
            return image, target, depth, normal, mask
        else:
            image = Image.open(self.images[idx])
            if self.transforms is not None:
                image = self.transforms(image)
            image = transforms.ToTensor()(image)

            return image

    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, target):
        target = target+1  # 255 -> 0, 0->1, 1->2
        return cls.cmap[target]