# Modified from https://github.com/VainF/nyuv2-python-toolkit
import os
import torch
import torch.utils.data as data
from PIL import Image
from scipy.io import loadmat
import numpy as np
import glob
from torchvision import transforms
from torchvision.datasets import VisionDataset
import random

from .utils import colormap

class NYUv2(VisionDataset):
    """NYUv2 dataset
    See https://github.com/VainF/nyuv2-python-toolkit for more details.
    
    Args:
        root (string): Root directory path.
        split (string, optional): 'train' for training set, and 'test' for test set. Default: 'train'.
        target_type (string, optional): Type of target to use, ``semantic``, ``depth`` or ``normal``. 
        num_classes (int, optional): The number of classes, must be 40 or 13. Default:13.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry and returns a transformed version.
    """
    cmap = colormap()
    def __init__(self,
                 root,
                 split='train',
                 target_type='semantic',
                 num_classes=13,
                 transforms=None,
                 transform=None,
                 target_transform=None):
        super( NYUv2, self ).__init__(root, transforms=transforms, transform=transform, target_transform=target_transform)
        assert(split in ('train', 'test'))

        self.root = root
        self.split = split
        self.target_type = target_type
        self.num_classes = num_classes
        
        split_mat = loadmat(os.path.join(self.root, 'splits.mat'))
        idxs = split_mat[self.split+'Ndxs'].reshape(-1) - 1

        img_names = os.listdir( os.path.join(self.root, 'image', self.split) )
        img_names.sort()
        images_dir = os.path.join(self.root, 'image', self.split)
        self.images = [os.path.join(images_dir, name) for name in img_names]

        self._is_depth = False
        if self.target_type=='semantic':
            semantic_dir = os.path.join(self.root, 'seg%d'%self.num_classes, self.split)
            self.labels = [os.path.join(semantic_dir, name) for name in img_names]
            self.targets = self.labels
        
        if self.target_type=='depth':
            depth_dir = os.path.join(self.root, 'depth', self.split)
            self.depths = [os.path.join(depth_dir, name) for name in img_names]
            self.targets = self.depths
            self._is_depth = True
        
        if self.target_type=='normal':
            normal_dir = os.path.join(self.root, 'normal', self.split)
            self.normals = [os.path.join(normal_dir, name) for name in img_names]
            self.targets = self.normals
        
    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        target = Image.open(self.targets[idx])
        if self.transforms is not None:
            image, target = self.transforms( image, target )
        return image, target

    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_fn(cls, mask: np.ndarray):
        """decode semantic mask to RGB image"""
        mask = mask.astype('uint8') + 1 # 255 => 0
        return cls.cmap[mask]
