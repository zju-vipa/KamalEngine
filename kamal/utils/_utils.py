import numpy as np
import math
import torch 
import random
from copy import deepcopy
import contextlib

def split_batch(batch):
    inputs, *targets = batch
    if len(targets)==1:
        targets = targets[0]
    return inputs, targets

@contextlib.contextmanager
def set_mode(model, training=True):
    ori_mode = model.training
    model.train(training)
    yield
    model.train(ori_mode)

def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device=device)
    elif isinstance( obj, (list, tuple) ):
        return [ o.to(device=device) for o in obj ]
    elif isinstance(obj, nn.Module):
        return obj.to(device=device)


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
    def __init__(self, mean, std, reverse=False):
        self.mean = mean
        self.std = std
        self.reverse = reverse

    def __call__(self, x):
        if self.reverse:
            return self.denormalize(x)
        else:
            return self.normalize(x)
            
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
                    _flatten( prefix+'%s.'%k, v )
            else:
                flattned[ (prefix+'%s.'%k).strip('.') ] = v
        
    _flatten('', dic)
    return flattned

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def count_parameters(model):
    return sum( [ p.numel() for p in model.parameters() ] )