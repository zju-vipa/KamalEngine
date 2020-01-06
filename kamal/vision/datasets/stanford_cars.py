import os
import glob
from PIL import Image
import numpy as np
from scipy.io import loadmat

from torch.utils import data
from .utils import download_url, mkdir

from shutil import copyfile


class StanfordCars(data.Dataset):
    """Dataset for Stanford Cars
    """

    urls = {'cars_train.tgz':     'http://imagenet.stanford.edu/internal/car196/cars_train.tgz',
            'cars_test.tgz':       'http://imagenet.stanford.edu/internal/car196/cars_test.tgz',
            'car_devkit.tgz':      'https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz',
            'cars_test_annos_withlabels.mat': 'http://imagenet.stanford.edu/internal/car196/cars_test_annos_withlabels.mat'}

    def __init__(self, root, split='train', download=False, transforms=None, offset=0):
        self.root = os.path.abspath(root)
        self.split = split
        self.transforms = transforms
        self.offset = offset

        if download:
            self.download()

        if self.split == 'train':
            annos = os.path.join(self.root, 'devkit', 'cars_train_annos.mat')
        else:
            annos = os.path.join(self.root, 'devkit',
                                 'cars_test_annos_withlabels.mat')

        annos = loadmat(annos)
        size = len(annos['annotations'][0])

        self.files = glob.glob(os.path.join(
            self.root, 'cars_'+self.split, '*.jpg'))
        self.files.sort()

        self.labels = np.array([int(l[4])-1 for l in annos['annotations'][0]])

        lbl_annos = loadmat(os.path.join(self.root, 'devkit', 'cars_meta.mat'))

        self.object_categories = [str(c[0])
                                  for c in lbl_annos['class_names'][0]]

        print('Stanford Cars, Split: %s, Size: %d' %
              (self.split, self.__len__()))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, 'Images',
                                      self.files[idx])).convert("RGB")
        lbl = self.labels[idx]
        if self.transforms is not None:
            img = self.transforms(img)
        return img, lbl+self.offset

    def download(self):
        import tarfile

        mkdir(self.root)
        for fname, url in self.urls.items():
            if not os.path.isfile(os.path.join(self.root, fname)):
                download_url(url, self.root, fname)
            if fname.endswith('tgz'):
                print("Extracting %s..." % fname)
                with tarfile.open(os.path.join(self.root, fname), "r:gz") as tar:
                    tar.extractall(path=self.root)

        copyfile(os.path.join(self.root, 'cars_test_annos_withlabels.mat'),
                 os.path.join(self.root, 'devkit', 'cars_test_annos_withlabels.mat'))
