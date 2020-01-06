import os
import torch.utils.data as data
from glob import glob
from PIL import Image
import numpy as np

def colormap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


class SUNRGBD(data.Dataset):
    """SUNRGBD dataset loader where the dataset is arranged as in https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid.
    
    **Parameters:**
        - **root_dir** (string): Root directory path.
        - **mode** (string): The type of dataset: 'train' for training set, 'val'. for validation set, and 'test' for test set.
        - **transform** (callable, optional): A function/transform that  takes in an PIL image and returns a transformed version. Default: None.
        - **label_transform** (callable, optional): A function/transform that takes in the target and transform it. Default: None.
        - **loader** (callable, optional): A function to load an image given its path. By default ``default_loader`` is used.
    """

    # Default encoding for pixel value, class name, and class color
    cmap = colormap()
    def __init__(self,
                 root,
                 split='train',
                 transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        self.images = glob(os.path.join(self.root, 'SUNRGBD-%s_images'%self.split, '*.jpg'))
        self.labels = glob(os.path.join(self.root, '%s13labels'%self.split, '*.png'))

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
        label = label-1  # ignore void 0->255
        return img, label

    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        #mask[mask == 255] = 11
        return cls.cmap[mask.astype('uint8')+1]