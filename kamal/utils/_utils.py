import numpy as np
import math
import torch 

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

def denormalize(tensor, mean, std):
    _mean = [ -m / s for m, s in zip(mean, std) ]
    _std = [ 1/s for s in std ]

    _mean = torch.as_tensor(_mean, dtype=tensor.dtype, device=tensor.device)
    _std = torch.as_tensor(_std, dtype=tensor.dtype, device=tensor.device)

    tensor = (tensor - _mean[None, :, None, None]) / (_std[None, :, None, None])
    return tensor

class Denormalizer(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return denormalize(x, self.mean, self.std)

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