import os
import json
from typing import Dict, Any, List, Tuple

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.folder import is_image_file, pil_loader


class TinyImagenet:
    """
    Image folder:

        train
            n01443537
                images
                n01443537_boxes.txt
            ...
        val
            images

            val_annotations.txt
                "val_9992.JPEG	n02231487	30	8	56	41"
    """
    def __init__(self, root: str, split: str, transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform

        self.meta_fp = os.path.join(self.root, "meta.json")
        meta = self._parse_dataset()
        self.classes = meta["classes"]
        self.class_to_idx = meta["class_to_idx"]
        self.words_map = meta["words_map"]
        if self.split == "train":
            self.samples = meta["train_samples"]
            self.targets = meta["train_targets"]
        elif self.split == "val":
            self.samples = meta["val_samples"]
            self.targets = meta["val_targets"]
        else:
            raise Exception("Wrong split type: {}, supported: `train`, `val`".format(self.split))

    def _parse_dataset(self) -> Dict[str, Any]:
        # read
        if os.path.isfile(self.meta_fp):
            with open(self.meta_fp, "r") as f:
                try:
                    meta = json.load(f)
                    return meta
                except:
                    pass

        train_path = os.path.join(self.root, "train")
        val_path = os.path.join(self.root, "val")

        meta: Dict[str, Any] = dict()

        # wid <==> class_idx
        classes, class_to_idx = self._find_classes(train_path)
        meta["classes"] = classes
        meta["class_to_idx"] = class_to_idx

        # read words
        meta["words_map"] = self._read_words(os.path.join(self.root, "words.txt"))

        # get samples
        train_samples = self._parse_train_images(train_path, classes, class_to_idx)
        val_samples = self._parse_val_images(val_path, classes, class_to_idx)

        if len(train_samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(train_path)
            raise RuntimeError(msg)
        if len(val_samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(val_path)
            raise RuntimeError(msg)

        meta["train_samples"] = train_samples
        meta["train_targets"] = [s[1] for s in train_samples]
        meta["val_samples"] = val_samples
        meta["val_targets"] = [s[1] for s in val_samples]

        with open(self.meta_fp, "w") as f:
            json.dump(meta, f)

        return meta

    def _find_classes(self, train_path):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes: List[str] = [d.name for d in os.scandir(train_path) if d.is_dir()]
        classes.sort()
        class_to_idx: Dict[str, int] = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def _read_words(self, words_filepath):
        words_map: Dict[str, str] = dict()
        with open(words_filepath, "r") as f:
            context = f.read()
        context = context.strip().split("\n")
        for line in context:
            line = line.strip().split("\t")
            wid = line[0]
            word = line[1]
            words_map[wid] = word
        return words_map

    def _parse_train_images(
        self,
        train_path: str,
        classes: List[str],
        class_to_idx: Dict[str, int]
    ) -> List[Tuple[str, int]]:
        instances = []
        for target_class in classes:
            target_dir = os.path.join(train_path, target_class, "images")
            class_index = class_to_idx[target_class]
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_image_file(path):
                        item = path, class_index
                        instances.append(item)
        return instances

    def _parse_val_images(
        self,
        val_path: str,
        classes: List[str],
        class_to_idx: Dict[str, int]
    ) -> List[Tuple[str, int]]:
        val_annotations_filepath = os.path.join(val_path, "val_annotations.txt")
        with open(val_annotations_filepath, "r") as f:
            context = f.read()
        context: List[str] = context.strip().split("\n")

        instances = []
        for line in context:
            annotation = line.strip().split("\t")
            img_filepath = os.path.join(val_path, "images", annotation[0])
            wid = annotation[1]
            class_index = class_to_idx[wid]
            if is_image_file(img_filepath):
                item = img_filepath, class_index
                instances.append(item)

        return instances

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.samples)


def get_tiny_imagenet(root: str, split: str = "train") -> Dataset:
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if split == "train":
        transform = train_transform
    else:
        transform = test_transform
    dataset = TinyImagenet(root, split=split, transform=transform)
    return dataset

