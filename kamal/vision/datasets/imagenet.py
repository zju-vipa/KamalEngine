import os
import shutil
from typing import Optional, Dict, Tuple
import tqdm

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.utils import check_integrity, verify_str_arg
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
from torchvision.datasets.folder import has_file_allowed_extension

META_FILE = "meta.bin"


class ImageNet(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        num_image_per_class: int = None,
        disable_parse_val: bool = False,
        transform=None,
        target_transform=None
    ):
        root = self.root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", ("train", "val"))

        self.disable_parse_val = disable_parse_val
        print("parsing archives...")
        self.parse_archives()
        wnid_to_classes = load_meta_file(self.root)[0]

        # start image folder {
        classes, class_to_idx = self._find_classes(self.split_folder)
        samples = make_dataset(
            directory=self.split_folder,
            class_to_idx=class_to_idx,
            num_image_per_class=num_image_per_class,
            extensions=IMG_EXTENSIONS
        )
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.split_folder + "\n"
                                "Supported extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.loader = default_loader

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        # }

        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx
                             for idx, clss in enumerate(self.classes)
                             for cls in clss}

        self.transform = transform
        self.target_transform = target_transform

    def parse_archives(self):
        if not check_integrity(os.path.join(self.root, META_FILE)):
            print("creating meta file...")
            parse_devkit_archive(self.root)
        print("parsing val archive...")
        parse_val_archive(self.root, disable=self.disable_parse_val)

    @property
    def split_folder(self):
        folder_map = dict(
            train="train_set",
            val="val_set"
        )
        return os.path.join(self.root, folder_map[self.split])

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)

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
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
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
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])


def load_meta_file(root, file=None):
    if file is None:
        file = META_FILE
    file = os.path.join(root, file)

    if check_integrity(file):
        return torch.load(file)
    else:
        msg = ("The meta file {} is not present in the root directory or is corrupted. "
               "This file is automatically created by the ImageNet dataset.")
        raise RuntimeError(msg.format(file, root))


def _verify_archive(root, file, md5):
    if not check_integrity(os.path.join(root, file), md5):
        msg = ("The archive {} is not present in the root directory or is corrupted. "
               "You need to download it externally and place it in {}.")
        raise RuntimeError(msg.format(file, root))


def parse_devkit_archive(root):
    """Parse the devkit archive of the ImageNet2012 classification dataset and save
    the meta information in a binary file.

    Args:
        root (str): Root directory containing the devkit archive
    """
    import scipy.io as sio

    def parse_meta_mat(devkit_root):
        metafile = os.path.join(devkit_root, "data", "meta.mat")
        meta = sio.loadmat(metafile, squeeze_me=True)["synsets"]
        nums_children = list(zip(*meta))[4]
        meta = [meta[idx] for idx, num_children in enumerate(nums_children)
                if num_children == 0]
        idcs, wnids, classes = list(zip(*meta))[:3]
        classes = [tuple(clss.split(", ")) for clss in classes]
        idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}
        wnid_to_classes = {wnid: clss for wnid, clss in zip(wnids, classes)}
        return idx_to_wnid, wnid_to_classes

    def parse_val_groundtruth_txt(devkit_root):
        file = os.path.join(
            devkit_root, "data", "ILSVRC2012_validation_ground_truth.txt")
        with open(file, "r") as txtfh:
            val_idcs = txtfh.readlines()
        return [int(val_idx) for val_idx in val_idcs]

    if not os.path.isfile(os.path.join(root, META_FILE)):
        devkit_root = os.path.join(root, "devkit")
        idx_to_wnid, wnid_to_classes = parse_meta_mat(devkit_root)
        val_idcs = parse_val_groundtruth_txt(devkit_root)
        val_wnids = [idx_to_wnid[idx] for idx in val_idcs]

        torch.save((wnid_to_classes, val_wnids), os.path.join(root, META_FILE))


def parse_val_archive(root, wnids=None, folder="val_set", disable=False):
    """Parse the validation images archive of the ImageNet2012 classification dataset
    and prepare it for usage with the ImageNet dataset.

    Args:
        root (str): Root directory containing the validation images archive
        wnids (list, optional): List of WordNet IDs of the validation images. If None
            is given, the IDs are loaded from the meta file in the root directory
        folder (str, optional): Optional name for validation images folder. Defaults to
            "val"
    """
    if disable:
        return

    val_root = os.path.join(root, folder)
    images = sorted([os.path.join(val_root, image) for image in os.listdir(val_root)])

    if wnids is None:
        wnids = load_meta_file(root)[1]

    for wnid in set(wnids):
        new_dir = os.path.join(val_root, wnid)
        if not os.path.isdir(new_dir):
            os.mkdir(new_dir)

    for wnid, img_file in tqdm.tqdm(zip(wnids, images)):
        if os.path.isfile(img_file):
            shutil.move(img_file, os.path.join(val_root, wnid, os.path.basename(img_file)))


def make_dataset(
    directory: str,
    class_to_idx: Dict[str, int],
    num_image_per_class: int,
    extensions: Tuple[str] = None,
    is_valid_file: Optional[bool] = None,
):
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames)[:num_image_per_class]:
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
    return instances


def get_imagenet(root: str, split: str = "train", num_image_per_class: int = None) -> Dataset:
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    if split == "train":
        transform = train_transform
    else:
        transform = test_transform
    dataset = ImageNet(root, split=split, transform=transform, num_image_per_class=num_image_per_class)
    return dataset

