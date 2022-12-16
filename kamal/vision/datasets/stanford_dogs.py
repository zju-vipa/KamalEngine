import os
import numpy as np
from PIL import Image
from scipy.io import loadmat

from torch.utils import data
from .utils import download_url
from shutil import move

class StanfordDogs(data.Dataset):
    """Dataset for Stanford Dogs
    """
    urls = {"images.tar":       "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar",
            "annotation.tar":   "http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar",
            "lists.tar":        "http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar"}

    def __init__(self, root, split='train', download=False, transform=None, target_transform=None):
        self.root = os.path.abspath( os.path.expanduser(root) )
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        if download:
            self.download()
        list_file = os.path.join(self.root, self.split+'_list.mat')
        mat_file = loadmat(list_file)
        size = len(mat_file['file_list'])
        self.files = [str(mat_file['file_list'][i][0][0]) for i in range(size)]
        self.labels = np.array(
            [mat_file['labels'][i][0]-1 for i in range(size)])
        categories = os.listdir(os.path.join(self.root, 'Images'))
        categories.sort()
        self.object_categories = [c[10:] for c in categories]
        print('Stanford Dogs, Split: %s, Size: %d' %
              (self.split, self.__len__()))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, 'Images',
                                      self.files[idx])).convert("RGB")
        lbl = self.labels[idx]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            lbl = self.target_transform( lbl )
        return img, lbl

    def download(self):
        import tarfile
        os.makedirs(self.root, exist_ok=True)
        for fname, url in self.urls.items():
            if not os.path.isfile(os.path.join(self.root, fname)):
                download_url(url, self.root, fname)
                # extract file
            print("Extracting %s..." % fname)
            with tarfile.open(os.path.join(self.root, fname), "r") as tar:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tar, path=self.root)
