import collections
import torch
import torchvision
import numpy as np
from PIL import Image
import os
from torchvision.datasets import VisionDataset
from .utils import colormap

class ADE20K(VisionDataset):
    cmap = colormap()

    def __init__(
        self,
        root,
        split="training",
        transform=None,
        target_transform=None,
        transforms=None,
    ):
        super( ADE20K, self ).__init__(  root=root, transforms=transforms, transform=transform, target_transform=target_transform )
        assert split in ['training', 'validation'], "split should be \'training\' or \'validation\'"
        self.root = os.path.expanduser(root)
        self.split = split
        self.num_classes = 150

        img_list = []
        lbl_list = []
        img_dir = os.path.join( self.root, 'images', self.split )
        lbl_dir = os.path.join( self.root, 'annotations', self.split )

        for img_name in os.listdir( img_dir ):
            img_list.append( os.path.join( img_dir, img_name ) )
            lbl_list.append( os.path.join( lbl_dir, img_name[:-3]+'png') )

        self.img_list = img_list
        self.lbl_list = lbl_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = Image.open( self.img_list[index] )
        lbl = Image.open( self.lbl_list[index] )
        if self.transforms:
            img, lbl = self.transforms(img, lbl)
            lbl = np.array(lbl, dtype='uint8')-1 # 1-150 => 0-149 + 255
        return img, lbl

    @classmethod
    def decode_seg_to_color(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask+1]
