import os

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.utils.data as data

from PIL import Image
import numpy as np
import os
import os.path
import sys


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


class DatasetFolderMultiSet(data.Dataset):

    def __init__(self, roots, loader, extensions, transform=None, target_transform=None):
        classes = list()
        class_to_idx = dict()
        samples = list()
        for root in roots:
            tmp_classes, tmp_class_to_idx = self._find_classes(root)
            current_cls_num = len(classes)

            if current_cls_num !=0:
                for key in tmp_class_to_idx.keys():
                    tmp_class_to_idx[key] += current_cls_num

            classes += tmp_classes
            class_to_idx.update(tmp_class_to_idx)

            tmp_samples = make_dataset(root, class_to_idx, extensions)
            if len(tmp_samples) == 0:
                raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                                    "Supported extensions are: " + ",".join(extensions)))
            samples += tmp_samples

        self.roots = roots
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

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
        fmt_str += '    Root Location: {}\n'.format(self.roots)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolderMultiSet(DatasetFolderMultiSet):

    def __init__(self, roots, transform=None, target_transform=None,
                 loader=default_loader):
        super(ImageFolderMultiSet, self).__init__(roots, loader, IMG_EXTENSIONS,
                                                   transform=transform,
                                                   target_transform=target_transform)
        self.imgs = self.samples


# ----------- Read Samples from Txt -----------
class DatasetFolderTxt(data.Dataset):

    def __init__(self, root, txt, loader, extensions, transform=None, target_transform=None):
        classes, class_to_idx = self._find_classes(root)
        self.class_to_idx = class_to_idx
        samples = self._complete_path(root, txt)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                                                                            "Supported extensions are: " + ",".join(
                extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform

    def _find_classes(self, dir):
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def _complete_path(self, root, txt):
        with open(txt, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]

            samples = []
            for line in lines:
                cls_name = line.split('/')[0]
                label = self.class_to_idx[cls_name]
                path = os.path.join(root, line)
                samples.append((path, label))

        return samples

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
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class ImageFolderTxt(DatasetFolderTxt):
    def __init__(self, root, txt, transform=None, target_transform=None,
                 loader=default_loader):
        super(ImageFolderTxt, self).__init__(root, txt, loader, IMG_EXTENSIONS,
                                             transform=transform,
                                             target_transform=target_transform)
        self.imgs = self.samples


class DatasetFolderPath(data.Dataset):

    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        classes, class_to_idx = self._find_classes(root)
        samples = make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                                                                            "Supported extensions are: " + ",".join(
                extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform

    def _find_classes(self, dir):
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class ImageFolderPath(DatasetFolderPath):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        super(ImageFolderPath, self).__init__(root, loader, IMG_EXTENSIONS,
                                              transform=transform,
                                              target_transform=target_transform)
        self.imgs = self.samples


def get_transform(img_set):
    if img_set == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.81, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    return transform


def get_dataloader(root_path, img_set, batch_size, shuffle, part_num=0, is_part=False):
    whole_path = os.path.join(root_path, 'images_whole')
    if is_part:
        path = os.path.join(root_path, 'image_part{}'.format(part_num), img_set)
        dataset = datasets.ImageFolder(root=path,
                                       transform=get_transform(img_set))
    else:
        print('Use whole data, whole path: {}'.format(whole_path))
        path = os.path.join(whole_path, img_set)
        dataset = datasets.ImageFolder(root=path,
                                       transform=get_transform(img_set))

    print('The size of dataset: {}'.format(len(dataset)))

    dataloder = DataLoader(dataset, batch_size=batch_size,
                           shuffle=shuffle, num_workers=6)

    return dataloder
def get_dataloader_path(root_path, img_set, batch_size, shuffle,
                        part_num=0, is_part=False):
    whole_path = os.path.join(root_path, 'images_whole')
    if is_part:
        path = os.path.join(root_path, 'image_part{}'.format(part_num), img_set)
        dataset = ImageFolderPath(root=path,
                                  transform=get_transform(img_set))
    else:
        print('Use whole data, whole path: {}'.format(whole_path))
        path = os.path.join(whole_path, img_set)
        dataset = ImageFolderPath(root=path,
                                  transform=get_transform(img_set))

    print('The size of dataset: {}'.format(len(dataset)))

    dataloder = DataLoader(dataset, batch_size=batch_size,
                           shuffle=shuffle, num_workers=6)

    return dataloder


def get_dataloader_multi_set(dataset_roots, img_set, batch_size, shuffle):

    print('Dataset pathes: {}'.format(dataset_roots))

    paths = [os.path.join(p, img_set) for p in dataset_roots]

    dataset = ImageFolderMultiSet(roots=paths,
                                  transform=get_transform(img_set))

    print('The size of dataset: {}'.format(len(dataset)))

    dataloder = DataLoader(dataset, batch_size=batch_size,
                           shuffle=shuffle, num_workers=6)

    return dataloder


def get_dataloader_txt(dataset_name, txt, img_set, batch_size, shuffle,
                       part_num=0):
    root_path = '/nfs/yxy/data/{}/'.format(dataset_name)
    path = os.path.join(root_path, 'image_part{}'.format(part_num), img_set)
    print('Dataset path: ', path)
    dataset = ImageFolderTxt(root=path, txt=txt,
                             transform=get_transform(img_set))

    print('The size of dataset: {}'.format(len(dataset)))

    dataloder = DataLoader(dataset, batch_size=batch_size,
                           shuffle=shuffle, num_workers=6)

    return dataloder

def get_divided_dataloader(data_dir,path,component_part,aux_parts, batch_size,img_set, shuffle):
    num_teacher = len(aux_parts)
    datasets = []
    len_datasets = []
    for t in aux_parts:
        txt_path = path + '/data-t{}.txt'.format(t)
        data_path = os.path.join(data_dir, 'image_part{}'.format(component_part), img_set)
        print('Dataset path: ', data_path)
        dataset = ImageFolderTxt(root=data_path, txt=txt_path,
                                transform=get_transform(img_set))
        datasets.append(dataset)
        
        len_dataset = len(dataset)
        assert (len_dataset >= batch_size)
        len_datasets.append(len_dataset)
        print('The size of SourceNet_{} dataset: {}'.format(t, len_dataset))
    len_max = np.argmax(len_datasets)
    num_iter = int(len_datasets[len_max]/batch_size)

    print('The largest dataset: SourceNet: {}'.format(aux_parts[len_max]))
    print('Iteration num: {}'.format(num_iter))

    dataloders = []
    batch_sizes = []
    for i in range(num_teacher):
        if i == len_max:
            bs = batch_size
            batch_sizes.append(bs)
        else:
            bs = int(len_datasets[i]/num_iter)
            batch_sizes.append(bs)

        if bs == 0:
            print('calculate batch size=0, assert batch_size>0')
            print('make batch size=0')
            bs = 1

        print('SourceNet_{} batch_size: {}'.format(aux_parts[i], bs))

        dataloder = DataLoader(datasets[i], batch_size=bs, shuffle=shuffle, num_workers=6)
        dataloders.append(dataloder)

    return dataloders

dataset_to_cls_num = {
    'dog': 120,
    'airplane': 100,
    'cub': 200,
    'car': 196
}

dataset_to_epoch = {
    'dog': [30, 81],
    'airplane': [92, 89],
    'cub': [84, 85],
    'car': [64, 97]
}