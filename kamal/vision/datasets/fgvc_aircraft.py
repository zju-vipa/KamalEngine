#Modified from https://github.com/pytorch/vision/pull/467/files 
from __future__ import print_function
import torch.utils.data as data
from torchvision.datasets.folder import pil_loader, accimage_loader, default_loader
from PIL import Image
import os
import numpy as np

from .utils import download_url, mkdir


def make_dataset(dir, image_ids, targets):
    assert(len(image_ids) == len(targets))
    images = []
    dir = os.path.expanduser(dir)
    for i in range(len(image_ids)):
        item = (os.path.join(dir, 'fgvc-aircraft-2013b', 'data', 'images',
                             '%s.jpg' % image_ids[i]), targets[i])
        images.append(item)
    return images


def find_classes(classes_file):
    # read classes file, separating out image IDs and class names
    image_ids = []
    targets = []
    f = open(classes_file, 'r')
    for line in f:
        split_line = line.split(' ')
        image_ids.append(split_line[0])
        targets.append(' '.join(split_line[1:]))
    f.close()

    # index class names
    classes = np.unique(targets)
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    targets = [class_to_idx[c] for c in targets]

    return (image_ids, targets, classes, class_to_idx)


class FGVCAircraft(data.Dataset):
    """`FGVC-Aircraft <http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft>`_ Dataset.
    Args:
        root (string): Root directory path to dataset.
        class_type (string, optional): The level of FGVC-Aircraft fine-grain classification
            to label data with (i.e., ``variant``, ``family``, or ``manufacturer``).
        transforms (callable, optional): A function/transforms that takes in a PIL image
            and returns a transformed version. E.g. ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transforms that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in the root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    url = 'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'
    class_types = ('variant', 'family', 'manufacturer')
    splits = ('train', 'val', 'trainval', 'test')

    def __init__(self, root, class_type='variant', split='train', transform=None,
                 target_transform=None, loader=default_loader, download=False):
        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                split, ', '.join(self.splits),
            ))
        if class_type not in self.class_types:
            raise ValueError('Class type "{}" not found. Valid class types are: {}'.format(
                class_type, ', '.join(self.class_types),
            ))

        self.root = root
        self.class_type = class_type
        self.split = split
        self.classes_file = os.path.join(self.root, 'fgvc-aircraft-2013b', 'data',
                                         'images_%s_%s.txt' % (self.class_type, self.split))
        if download:
            self.download()

        (image_ids, targets, classes, class_to_idx) = find_classes(self.classes_file)
        samples = make_dataset(self.root, image_ids, targets)

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx

        with open(os.path.join(self.root, 'fgvc-aircraft-2013b/data', 'variants.txt')) as f:
            self.object_categories = [
                line.strip('\n') for line in f.readlines()]
        print('FGVC-Aircraft, Split: %s, Size: %d' % (self.split, self.__len__()))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transforms.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp, self.target_transforms.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, 'data', 'images')) and \
            os.path.exists(self.classes_file)

    def download(self):
        """Download the FGVC-Aircraft data if it doesn't exist already."""
        from six.moves import urllib
        import tarfile

        mkdir(self.root)

        fpath = os.path.join(self.root, 'fgvc-aircraft-2013b.tar.gz')
        if not os.path.isfile(fpath):
            download_url(self.url, self.root, 'fgvc-aircraft-2013b.tar.gz')
        print("Extracting fgvc-aircraft-2013b.tar.gz...")
        with tarfile.open(fpath, "r:gz") as tar:
            tar.extractall(path=self.root)
