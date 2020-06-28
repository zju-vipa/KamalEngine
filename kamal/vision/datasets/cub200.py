import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from .utils import download_url
from torch.utils.data import Dataset
import shutil


class CUB200(Dataset):
    base_folder = 'images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, split='train', transform=None, target_transform=None, loader=default_loader, download=False):
        self.root = os.path.abspath( os.path.expanduser( root ) )
        self.transform = transform
        self.target_transform = target_transform
        self.loader = default_loader
        self.split = split

        if download:
            self.download()
        self._load_metadata()
        categories = os.listdir(os.path.join(
            self.root, 'CUB_200_2011', 'images'))
        categories.sort()
        self.object_categories = [c[4:] for c in categories]
        print('CUB200, Split: %s, Size: %d' % (self.split, self.__len__()))

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.split == 'train':
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def download(self):
        import tarfile
        os.makedirs(self.root, exist_ok=True)
        if not os.path.isfile(os.path.join(self.root, self.filename)):
            download_url(self.url, self.root, self.filename)
        print("Extracting %s..." % self.filename)
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, 'CUB_200_2011',
                            self.base_folder, sample.filepath)
        lbl = sample.target - 1
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            lbl = self.target_transform(lbl)
        return img, lbl
