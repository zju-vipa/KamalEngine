import torch
from torch.utils.data import Dataset
import os

from PIL import Image
import random
from copy import deepcopy

def _collect_all_images(root, postfix=['png', 'jpg', 'jpeg', 'JPEG']):
    images = []
    if isinstance( postfix, str):
        postfix = [ postfix ]
    for dirpath, dirnames, files in os.walk(root):
        for pos in postfix:
            for f in files:
                if f.endswith( pos ):
                    images.append( os.path.join( dirpath, f ) )
    return images

def get_train_val_set(root, val_size=0.3):
    if not isinstance(root, (list, tuple)):
        root = [root]
    train_set = []
    val_set = []
    for _root in root:
        _part_train_set = _collect_all_images( _root)
        if os.path.isdir( os.path.join(_root, 'test') ):
            _part_val_set = _collect_all_images( os.path.join(_root, 'test') )
        else:
            _val_size = int( len(_part_train_set) * val_size )
            _part_val_set = random.sample( _part_train_set, k=_val_size )
        _part_train_set = [ d for d in _part_train_set if d not in _part_val_set ]
        train_set.extend(_part_train_set)
        val_set.extend(_part_val_set)
    return train_set, val_set

class UnlabeledDataset(Dataset):
    def __init__(self, data, transform=None, postfix=['png', 'jpg', 'jpeg', 'JPEG']):
        self.transform = transform
        self.data = data

    def __getitem__(self, idx):
        data = Image.open( self.data[idx] )
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.data)

