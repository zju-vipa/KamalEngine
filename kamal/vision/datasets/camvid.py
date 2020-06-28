import os
import torch.utils.data as data
from glob import glob
from PIL import Image
import numpy as np
from torchvision.datasets import VisionDataset

class CamVid(VisionDataset):
    """CamVid dataset loader where the dataset is arranged as in https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid.
    
    Args:
        root (string): 
        split (string): The type of dataset: 'train', 'val', 'trainval', or 'test'
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version. Default: None.
        target_transform (callable, optional): A function/transform that takes in the target and transform it. Default: None.
        transforms (callable, optional): A function/transform that takes in both the image and target and transform them. Default: None.
    """

    cmap = np.array([
        (128, 128, 128),
        (128, 0, 0),
        (192, 192, 128),
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
                 transform=None,
                 target_transform=None,
                 transforms=None):
        assert split in ('train', 'val', 'test', 'trainval')
        super( CamVid, self ).__init__(root=root, transforms=transforms, transform=transform, target_transform=target_transform)
        self.root = os.path.expanduser(root)
        self.split = split

        if split == 'trainval':
            self.images = glob(os.path.join(self.root, 'train', '*.png')) + glob(os.path.join(self.root, 'val', '*.png'))
            self.labels = glob(os.path.join(self.root, 'trainannot', '*.png')) + glob(os.path.join(self.root, 'valannot', '*.png'))
        else:
            self.images = glob(os.path.join(self.root, self.split, '*.png'))
            self.labels = glob(os.path.join(self.root, self.split+'annot', '*.png'))
            
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
        if self.transforms is not None:
            img, label = self.transforms(img, label)
        label[label == 11] = 255  # ignore void
        return img, label.squeeze(0)
    
    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_fn(cls, mask):
        """decode semantic mask to RGB image"""
        mask[mask == 255] = 11
        return cls.cmap[mask]