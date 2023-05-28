# Copyright 2020 Zhejiang Lab. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================

import numpy as np
import math
import torch 
import random
from copy import deepcopy
import contextlib, hashlib
from contextlib import contextmanager
import os
from PIL import Image

def split_batch(batch):
    if isinstance(batch, (list, tuple)):
        inputs, *targets = batch
        if len(targets)==1:
            targets = targets[0]
        return inputs, targets
    else:
        return [batch, None] 

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
        return normalize( x, self.mean, self.std)
    
    def denormalize(self, x):
        return normalize( x, self.mean, self.std)


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

@contextmanager
def dummy_ctx(*args, **kwds):
    try:
        yield None
    finally:
        pass
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

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

class DataIter(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._iter = iter(self.dataloader)
    
    def next(self):
        try:
            data = next( self._iter )
        except StopIteration:
            self._iter = iter(self.dataloader)
            data = next( self._iter )
        return data

def save_image_batch(imgs, output, col=None, size=None, pack=True):
    if isinstance(imgs, torch.Tensor):
        imgs = (imgs.detach().clamp(0, 1).cpu().numpy()*255).astype('uint8')
    base_dir = os.path.dirname(output)
    if base_dir!='':
        os.makedirs(base_dir, exist_ok=True)
    if pack:
        imgs = pack_images( imgs, col=col ).transpose( 1, 2, 0 ).squeeze()
        imgs = Image.fromarray( imgs )
        if size is not None:
            if isinstance(size, (list,tuple)):
                imgs = imgs.resize(size)
            else:
                w, h = imgs.size
                max_side = max( h, w )
                scale = float(size) / float(max_side)
                _w, _h = int(w*scale), int(h*scale)
                imgs = imgs.resize([_w, _h])
        imgs.save(output)
    else:
        output_filename = output.strip('.png')
        for idx, img in enumerate(imgs):
            img = Image.fromarray( img.transpose(1, 2, 0) )
            img.save(output_filename+'-%d.png'%(idx))

def prepare_ood_subset(ood_dst, ood_size, teachers):
    ood_loader = torch.utils.data.DataLoader(ood_dst, batch_size=2048, shuffle=False, num_workers=4)
    entropy_list = []

    with set_mode(teachers, training=False):
        with torch.no_grad():
            for image, _ in tqdm(ood_loader):
                image = image.cuda()

                t_out = [teacher(image) for teacher in teachers]
                pyx = [F.softmax(i, dim=1) for i in t_out]
                entropy = sum([-(i * torch.log(i)).sum(dim=1) for i in pyx])

                entropy_list.append(entropy)

    entropy_list = torch.cat(entropy_list, dim=0)
    ood_index = torch.argsort(entropy_list, descending=True)[:ood_size].cpu().tolist()

    return ood_index