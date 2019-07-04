# coding:utf-8
import torch
from torch.utils.data.dataset import Dataset  # For custom data-sets
from torchvision import transforms

from PIL import Image
import glob
import os
import numpy as np
import random


class NYUDataset(Dataset):
    def __init__(self, root, train, transform):
        super(NYUDataset, self).__init__()
        self.transform = transform
        prefix = 'train' if train else 'test'
        self.folder_path = os.path.join(root, prefix)
        self.names = [name[-8:-4]
                      for name in glob.glob(os.path.join(self.folder_path, 'IMAGE/*.jpg'))]
        self.names.sort()
        self.length = len(self.names)

    def __getitem__(self, index):
        assert(index >= 0)
        assert(index < self.length)

        image = Image.open(
            os.path.join(self.folder_path, 'IMAGE/{}.jpg'.format(self.names[index])))

        segmentation = Image.open(
            os.path.join(self.folder_path, 'SEGMENTATION/{}.png'.format(self.names[index])))

        depth = Image.open(
            os.path.join(self.folder_path, 'DEPTH/{}.png'.format(self.names[index])))

        normal = Image.open(
            os.path.join(self.folder_path, 'NORMAL/{}.png'.format(self.names[index])))

        mask = Image.open(
            os.path.join(self.folder_path, 'MASK/{}.png'.format(self.names[index])))
        mask = mask.convert('L')

        if transforms is not None:
            seed = np.random.randint(2147483647)
            random.seed(seed)
            image = self.transform(image)

            random.seed(seed)
            segmentation = self.transform(segmentation)

            random.seed(seed)
            depth = self.transform(depth)

            random.seed(seed)
            normal = self.transform(normal)

            random.seed(seed)
            mask = self.transform(mask)

        sample = {'image': image, 'segmentation': segmentation, 'depth': depth,
                  'normal': normal}
        return sample

    def __len__(self):
        return self.length
