import os
import torch
import torch.utils.data as data
from PIL import Image
from .utils import colormap, set_seed
import numpy as np


class Kitti(data.Dataset):
    """Kitti dataset loader.
    
    **Parameters:**
        - **root** (string): Root directory path.
        - **split** (string, optional): 'train' for training set, and 'test' for test set. Default: 'train'.
        - **transform** (callable, optional): A function/transform that takes in an PIL image and returns a transformed version. Default: None.
        - **target_transforms** (callable, optional): A function/transform that takes in the target and transform it. Default: None.
        - **ds_type** (string, optional): To pick samples with semantic labels, depth, or nothing. Default: 'seg'.
    """
    def __init__(self,
                 root,
                 split='train',
                 transforms=None,
                 target_transforms=None,
                 ds_type='seg'):

        assert(split in ('train', 'test'))
        assert(ds_type in ('seg', 'depth', 'unlabeled'))
        self.root = root
        self.split = split
        self.transforms = transforms
        self.ds_type = ds_type

        if ds_type == 'unlabeled':
            with open(os.path.join(root, 'raw_data/unlabeled.txt')) as f:
                self.images = f.readlines()

        else:
            self.images = []
            self.targets = []
            with open(os.path.join(root, '{}/{}.txt'.format('semantics' if ds_type == 'seg' else 'depths', split))) as f:
                for line in f:
                    names = line.split
                    self.images.append(names[0])
                    self.targets.append(names[1])

    def __getitem__(self, idx):
        if self.ds_type == 'unlabeled':
            image = Image.open(self.images[idx])
            if self.transforms:
                image = self.transforms(image)
            return image
        else:
            image = Image.open(self.images[idx])
            target = Image.open(self.targets[idx])

            seed = np.random.randint(2147483647)
            if self.transforms:
                set_seed(seed)
                image = self.transforms(image)

            if self.target_transforms:
                set_seed(seed)
                target = self.target_transforms(target)

            return image, target

    def __len__(self):
        return len(self.images)
