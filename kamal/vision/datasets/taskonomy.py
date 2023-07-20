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
import torchvision.transforms.functional as TF
import random
import lmdb
import ipdb
import pickle


class Taskonomy(VisionDataset):
    """Taskonomy dataset
    
    Args:
        root (string): Root directory path.
        split (string, optional): 'teacher' for teacher net,'train' for training set, and 'test' for test set. Default: 'train'.
        target_type (string, optional): Type of target to use, 'normal', 'edge_texture', 'edge_occlusion', 'depth_euclidean', 'keypoints3d'. 
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry and returns a transformed version.
    """
    def __init__(self,
                 root,
                 split='train',
                 target_type1='normal',
                 target_type2=None,
                 num_classes=13,
                 transforms=None,
                 transform=None,
                 target_transform=None):
        super( Taskonomy, self ).__init__(root, transforms=transforms, transform=transform, target_transform=target_transform)
        assert(split in ('train', 'test'))
        assert(target_type1 in ('normal', 'edge_texture', 'edge_occlusion', 'depth_euclidean', 'keypoints3d'))
        if target_type2 != None:
            assert(target_type2 in ('normal', 'edge_texture', 'edge_occlusion', 'depth_euclidean', 'keypoints3d'))

        self.root = root
        self.split = split
        self.target_type1 = target_type1
        self.target_type2 = target_type2
        
        #split_mat = loadmat(os.path.join(self.root, 'splits.mat'))
        #idxs = split_mat[self.split+'Ndxs'].reshape(-1) - 1
        
        file_names = os.listdir( os.path.join(self.root, "wiconisco", 'rgb', self.split) )
        self.img_names = [filename.split('_rgb')[0] for filename in file_names]
        # img_names.sort()
        images_dir = os.path.join(self.root, "wiconisco", 'rgb', self.split)
        self.images = [os.path.join(images_dir, name) for name in file_names]

        self._is_depth = False

        
        if self.target_type1=='depth_euclidean':
            depth_dir = os.path.join(self.root, "wiconisco", 'depth_euclidean')
            self.depths = [os.path.join(depth_dir, '{}_depth_euclidean.png'.format(name)) for name in self.img_names]
            self.targets1 = self.depths
            self._is_depth = True
        
        if self.target_type1=='normal':
            normal_dir = os.path.join(self.root, "wiconisco", 'normal')
            self.normals = [os.path.join(normal_dir, '{}_normal.png'.format(name)) for name in self.img_names]
            self.targets1 = self.normals
        
        if self.target_type1=='edge_texture':
            normal_dir = os.path.join(self.root, "wiconisco", 'edge_texture')
            self.edge2d = [os.path.join(normal_dir, '{}_edge_texture.png'.format(name)) for name in self.img_names]
            self.targets1 = self.edge2d

        if self.target_type1=='edge_occlusion':
            normal_dir = os.path.join(self.root, "wiconisco", 'edge_occlusion')
            self.edge3d = [os.path.join(normal_dir, '{}_edge_occlusion.png'.format(name)) for name in self.img_names]
            self.targets1 = self.edge3d

        if self.target_type2=='depth_euclidean':
            depth_dir = os.path.join(self.root, "wiconisco", 'depth_euclidean')
            self.depths = [os.path.join(depth_dir, '{}_depth_euclidean.png'.format(name)) for name in self.img_names]
            self.targets2 = self.depths
            self._is_depth = True
        
        if self.target_type2=='normal':
            normal_dir = os.path.join(self.root, "wiconisco", 'normal')
            self.normals = [os.path.join(normal_dir, '{}_normal.png'.format(name)) for name in self.img_names]
            self.targets2 = self.normals
        
        if self.target_type2=='edge_texture':
            normal_dir = os.path.join(self.root, "wiconisco", 'edge_texture')
            self.edge2d = [os.path.join(normal_dir, '{}_edge_texture.png'.format(name)) for name in self.img_names]
            self.targets2 = self.edge2d

        if self.target_type2=='edge_occlusion':
            normal_dir = os.path.join(self.root, "wiconisco", 'edge_occlusion')
            self.edge3d = [os.path.join(normal_dir, '{}_edge_occlusion.png'.format(name)) for name in self.img_names]
            self.targets2 = self.edge3d
        
    def __getitem__(self, idx):
        env1 = lmdb.open(os.path.join(self.root, 'database', self.target_type1))
        txn1 = env1.begin()
        if self.target_type2 != None:
            env2 = lmdb.open(os.path.join(self.root, 'database', self.target_type2))
            txn2 = env2.begin()
        image_ori = Image.open(self.images[idx])
        image = TF.to_tensor(TF.resize(image_ori, 256)) * 2 - 1
        # image = image.unsqueeze_(0)
        target_true_ori = Image.open(self.targets1[idx])
        if self.target_type1 in ('edge_occlusion', 'edge_texture', 'depth_euclidean'):
            # target_true = TF.to_tensor(TF.resize(target_true_ori, 256)).float() / 65535 * 2 - 1
            target_true = TF.to_tensor(TF.resize(target_true_ori, 256)).float()
            # ipdb.set_trace()
        else:
            target_true = TF.to_tensor(TF.resize(target_true_ori, 256)) * 256
            # target_true = TF.to_tensor(TF.resize(target_true_ori, 256)) * 2 - 1
      

       
        
        if self.split == 'train':
            value1 = txn1.get(self.img_names[idx].encode())
      
            target_soft1, encoder_rep1, encoder_mia1 = pickle.loads(value1)
            if self.target_type2 != None:
                value2 = txn2.get(self.img_names[idx].encode())
                target_soft2, encoder_rep2, encoder_mia2 = pickle.loads(value2)
                t_data = (target_soft1, target_soft2, encoder_mia1, encoder_mia2, encoder_rep1, encoder_rep2)
      
                return image, t_data
            t_data = (target_soft1, encoder_rep1, encoder_mia1)
            return image, t_data
        else:
            return image, target_true

    def __len__(self):
        return len(self.images)

