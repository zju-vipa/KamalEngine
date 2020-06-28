import numpy as np
import math
import torch 
import random
from copy import deepcopy

from ruamel.yaml import YAML

import contextlib

@contextlib.contextmanager
def set_mode(model, training=True):
    ori_mode = model.training
    model.train(training)
    yield
    model.train(ori_mode)
    
def save_yaml(dic: dict, filepath: str):
    yaml=YAML()  
    with open(filepath, 'w') as f:
        yaml.dump(doc, f)

def load_yaml(filepath):
    yaml=YAML()  
    with open(filepath, 'r') as f:
        return yaml.load(f)
    
def pack_images(images, col=None, channel_last=False):
    # N, C, H, W
    if isinstance(images, (list, tuple) ):
        images = np.stack(images, 0)
    if channel_last:
        images = images.transpose(0,3,1,2) # make it channel first
    assert len(images.shape)==4
    assert isinstance(images, np.ndarray)
    
    N,C,H,W = images.shape
    if col is None:
        col = int(math.ceil(math.sqrt(N)))
    row = int(math.ceil(N / col))
    pack = np.zeros( (C, H*row, W*col), dtype=images.dtype )
    for idx, img in enumerate(images):
        h = (idx//col) * H
        w = (idx% col) * W
        pack[:, h:h+H, w:w+W] = img
    return pack

def normalize(tensor, mean, std, reverse=False):
    if reverse:
        _mean = [ -m / s for m, s in zip(mean, std) ]
        _std = [ 1/s for s in std ]
    else:
        _mean = mean
        _std = std
    _mean = torch.as_tensor(_mean, dtype=tensor.dtype, device=tensor.device)
    _std = torch.as_tensor(_std, dtype=tensor.dtype, device=tensor.device)
    tensor = (tensor - _mean[None, :, None, None]) / (_std[None, :, None, None])
    return tensor

class Normalizer(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def normalize(self, x):
        return normalize( x, self.mean, self.std )
    
    def denormalize(self, x):
        return normalize( x, self.mean, self.std, reverse=True )


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

DEFAULT_COLORMAP = colormap()

def flatten_dict(dic):
    flattned = dict()

    def _flatten(prefix, d):
        for k, v in d.items():
            if isinstance(v, dict):
                if prefix is None:
                    _flatten( k, v )
                else:
                    _flatten( prefix+'%s/'%k, v )
            else:
                flattned[ (prefix+'%s/'%k).strip('/') ] = v
        
    _flatten('', dic)
    return flattned

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def count_parameters(model):
    return sum( [ p.numel() for p in model.parameters() ] )

class TaskSplitter():
    def __init__(self, dataset, class_split, data_attr, targets_attr):
        self.class_split = class_split
        data = getattr( dataset, data_attr)
        targets = getattr( dataset, targets_attr )
        n_parts = len(class_split)
        self.dataset_split = [ deepcopy( dataset ) for _ in range(n_parts) ]
        
        for p in range(n_parts):
            setattr( self.dataset_split[p], data_attr, [] )
            setattr( self.dataset_split[p], targets_attr, [] )
            data_p = getattr( self.dataset_split[p], data_attr)
            targets_p = getattr( self.dataset_split[p], targets_attr)
            for new_c, c in enumerate(self.class_split[p]):
                for i, (d, t) in enumerate( zip(data, targets) ):
                    if t==c:
                        data_p.append( d )
                        targets_p.append( new_c )
        
            if isinstance( data, np.ndarray ):
                setattr( self.dataset_split[p], data_attr, np.array(data_p) )
            if isinstance( targets, np.ndarray ):
                setattr( self.dataset_split[p], targets_attr, np.array(targets_p) )

    @staticmethod
    def create_class_split(num_classes, n_parts):
        class_list = np.random.permutation( num_classes )
        class_split = np.array_split( class_list, n_parts )
        class_split = [ l.tolist() for l in class_split ]
        return class_split

    def get_class_split(self):
        return self.class_split

    def get_dataset_split(self):
        return self.dataset_split

    def get_class_mapping(self, parts: list):
        mapping = np.concatenate( [ self.class_split[p] for p in parts ], axis=0)
        return mapping

from torch.utils.data import ConcatDataset
class TaskMerger(ConcatDataset):
    def __init__(self, datasets, num_classes: list):
        super( TaskMerger, self ).__init__(datasets)
        offset.insert( 0, 0 )
        self.offset = num_classes

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        data, target = self.datasets[dataset_idx][sample_idx]
        return data, target+self.offset[ dataset_idx ]