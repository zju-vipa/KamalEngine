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

class UnlabeledDataset(Dataset):
    def __init__(self, root, transform=None, postfix=['png', 'jpg', 'jpeg', 'JPEG']):
        self.root = root
        self.transform = transform
        if isinstance(root, (list, tuple)):
            self.data = []
            for _root in self.root:
                data = _collect_all_images( _root, postfix=postfix )
                self.data.extend( data )
        else:
            self.data = _collect_all_images( self.root, postfix=postfix  )

    def __getitem__(self, idx):
        data = Image.open( self.data[idx] )
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.data)

    def split(self, lengths):
        indexs = list( range(len(self.data)) )
        random.shuffle( indexs )
        splited = []
        ptr = 0
        for length in lengths:
            dst = deepcopy( self )
            part_data = [ self.data[i] for i in indexs[ ptr: ptr+length ] ]
            dst.data = part_data
            ptr+=length
            splited.append(dst)
        return splited


