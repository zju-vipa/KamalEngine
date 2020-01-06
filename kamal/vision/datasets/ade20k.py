import collections
import torch
import torchvision
import numpy as np
from PIL import Image
import os
from torchvision.datasets import VisionDataset
from glob import glob

class ADE20K(VisionDataset):
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

        #self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        #self.mean = np.array([104.00699, 116.66877, 122.67892])
        #self.files = collections.defaultdict(list)
        self.img_list = glob(os.path.join( self.root, 'images', self.split, '**', '*.jpg' ), recursive=True)
        self.img_list.sort()
        self.lbl_list = [ (p[:-4]+'.png').replace( 'images', 'annotations' ) for p in self.img_list  ]

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img = Image.open( self.img_list[index] )
        lbl = Image.open( self.lbl_list[index] )
        if self.transforms:
            img, lbl = self.transforms(img, lbl)
            lbl = np.array(lbl, dtype='uint8') - 1 # 1-150 => 0-149 + 255
        return img, lbl

    def decode_target(self, temp, plot=False):
        # from @meetshah1995
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = 10 * (l % 10)
            g[temp == l] = l
            b[temp == l] = 0
        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

if __name__ == "__main__":
    dst = ADE20K('~/Datasets/ADEChallengeData2016/')
    print(dst[0])