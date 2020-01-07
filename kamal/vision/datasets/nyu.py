#coding:utf-8
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

class NYUv2(VisionDataset):
    """NYUv2 dataset
    
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
        self.train_idx = np.array([255, ] + list(range(num_classes)))
        
        split_mat = loadmat(os.path.join(self.root, 'splits.mat'))
        idxs = split_mat[self.split+'Ndxs'].reshape(-1) - 1

        img_names = os.listdir( os.path.join(self.root, 'image', self.split) )
        img_names.sort()
        images_dir = os.path.join(self.root, 'image', self.split)
        self.images = [os.path.join(images_dir, name) for name in img_names]

        if self.target_type=='semantic':
            semantic_dir = os.path.join(self.root, 'seg%d'%self.num_classes, self.split)
            self.labels = [os.path.join(semantic_dir, name) for name in img_names]
            self.targets = self.labels

        if self.target_type=='depth':
            depth_dir = os.path.join(self.root, 'depth', self.split)
            self.depths = [os.path.join(depth_dir, name) for name in img_names]
            self.targets = self.depths
        
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

if __name__=='__main__':
    from torchvision import transforms
    import matplotlib.pyplot as plt
    nyu_semantic13 = NYUv2( root='NYUv2', split='train', target_type='semantic', num_classes=13, 
                            transform=transforms.Compose([
                                transforms.Resize(512),
                                transforms.ToTensor()
                            ]),
                            target_transform=transforms.Compose([
                                transforms.Resize(512, interpolation=Image.NEAREST),
                                transforms.Lambda(lambda lbl: torch.from_numpy( np.array(lbl, dtype='uint8')-1 ) ) # 0->255, 1->0, 2->1
                            ]),  
                        )

    nyu_semantic40 = NYUv2( root='NYUv2', split='train', target_type='semantic', num_classes=40, 
                            transform=transforms.Compose([
                                transforms.Resize(512),
                                transforms.ToTensor()
                            ]),
                            target_transform=transforms.Compose([
                                transforms.Resize(512, interpolation=Image.NEAREST),
                                transforms.Lambda(lambda lbl: torch.from_numpy( np.array(lbl, dtype='uint8')-1 ) ) # 0->255, 1->0, 2->1
                            ]),  
                        )

    nyu_depth = NYUv2( root='NYUv2', split='train', target_type='depth', 
                            transform=transforms.Compose([
                                transforms.Resize(512),
                                transforms.ToTensor()
                            ]),
                            target_transform=transforms.Compose([
                                transforms.Resize(512),
                                transforms.Lambda(lambda lbl: torch.from_numpy( np.array(lbl, dtype='float') )/1e3 ) # uint16 to depth
                            ]),  
                        )

    nyu_normal = NYUv2( root='NYUv2', split='train', target_type='normal', 
                            transform=transforms.Compose([
                                transforms.Resize(512),
                                transforms.ToTensor()
                            ]),
                            target_transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Lambda(lambda normal: normal * 2 - 1)
                            ]),  
                        )
        
    os.makedirs('test', exist_ok=True)
    # Semantic
    img_id = 0
    
    img, lbl13 = nyu_semantic13[img_id]
    Image.fromarray((img*255).numpy().transpose( 1,2,0 ).astype('uint8')).save('test/image.png')
    Image.fromarray( nyu_semantic13.cmap[ (lbl13.numpy().astype('uint8')+1) ] ).save('test/semantic13.png')

    img, lbl40 = nyu_semantic40[img_id]
    Image.fromarray( nyu_semantic40.cmap[ (lbl40.numpy().astype('uint8')+1) ] ).save('test/semantic40.png')

    # Depth
    img, depth = nyu_depth[img_id]
    norm = plt.Normalize()
    depth = plt.cm.jet(norm(depth))
    plt.imsave('test/depth.png', depth)

    # Normal
    img, normal = nyu_normal[img_id]
    normal = (normal+1)/2
    Image.fromarray((normal*255).numpy().transpose( 1,2,0 ).astype('uint8')).save('test/normal.png')

    