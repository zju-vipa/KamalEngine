import os
import torch.utils.data as data
from glob import glob
from PIL import Image
import numpy as np


class CamVid(data.Dataset):
    """CamVid dataset loader where the dataset is arranged as in https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid.
    
    **Parameters:**
        - **root_dir** (string): Root directory path.
        - **mode** (string): The type of dataset: 'train' for training set, 'val'. for validation set, and 'test' for test set.
        - **transform** (callable, optional): A function/transform that  takes in an PIL image and returns a transformed version. Default: None.
        - **label_transform** (callable, optional): A function/transform that takes in the target and transform it. Default: None.
        - **loader** (callable, optional): A function to load an image given its path. By default ``default_loader`` is used.
    """

    # Default encoding for pixel value, class name, and class color
    cmap = np.array([
        (128, 128, 128),
        (128, 0, 0),
        (192, 192, 128),
        #(255, 69, 0),
        (128, 64, 128),
        (60, 40, 222),
        (128, 128, 0),
        (192, 128, 128),
        (64, 64, 128),
        (64, 0, 128),
        (64, 64, 0),
        (0, 128, 192),
        (0, 0, 0),
    ])

    def __init__(self,
                 root,
                 split='train',
                 transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        self.images = glob(os.path.join(self.root, self.split, '*.png'))
        self.labels = glob(os.path.join(
            self.root, self.split+'_labels', '*.png'))
        self.images.sort()
        self.labels.sort()

    def __getitem__(self, idx):
        """
        Args:
        - index (``int``): index of the item in the dataset
        Returns:
        A tuple of ``PIL.Image`` (image, label) where label is the ground-truth
        of the image.
        """

        img, label = Image.open(self.images[idx]), Image.open(self.labels[idx])

        if self.transform is not None:
            img, label = self.transform(img, label)
        label[label == 11] = 255  # ignore void
        return img, label

    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        mask[mask == 255] = 11
        return cls.cmap[mask]
