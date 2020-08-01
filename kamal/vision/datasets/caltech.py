# Modified from https://github.com/pytorch/vision/blob/master/torchvision/datasets/caltech.py
from __future__ import print_function
from PIL import Image
import os
import os.path

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_url


class Caltech101(VisionDataset):
    """`Caltech 101 <http://www.vision.caltech.edu/Image_Datasets/Caltech101/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``caltech101`` exists or will be saved to if download is set to True.
        target_type (string or list, optional): Type of target to use, ``category`` or
        ``annotation``. Can also be a list to output a tuple with all specified target types.
        ``category`` represents the target class, and ``annotation`` is a list of points
        from a hand-generated outline. Defaults to ``category``.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(self, root, target_type="category", train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(Caltech101, self).__init__(os.path.join(root, 'caltech101'))
        self.train = train
        self.dir_name = '101_ObjectCategories_split/train' if self.train else '101_ObjectCategories_split/test'
        
        os.makdirs(self.root, exist_ok=True)
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.categories = sorted(os.listdir(os.path.join(self.root, "101_ObjectCategories")))
        self.categories.remove("BACKGROUND_Google")  # this is not a real class

        # For some reason, the category names in "101_ObjectCategories" and
        # "Annotations" do not always match. This is a manual map between the
        # two. Defaults to using same name, since most names are fine.
        name_map = {"Faces": "Faces_2",
                    "Faces_easy": "Faces_3",
                    "Motorbikes": "Motorbikes_16",
                    "airplanes": "Airplanes_Side_2"}
        self.annotation_categories = list(map(lambda x: name_map[x] if x in name_map else x, self.categories))

        self.index = []
        self.y = []
        for (i, c) in enumerate(self.categories):
            file_names = os.listdir(os.path.join(self.root, self.dir_name, c))
            n = len(file_names)
            self.index.extend( file_names )
            self.y.extend(n * [i])
            
        print(self.train, len(self.index))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where the type of target specified by target_type.
        """
        import scipy.io

        img = Image.open(os.path.join(self.root,
                                      self.dir_name,
                                      self.categories[self.y[index]],
                                      self.index[index])).convert("RGB")
        target = []
        for t in self.target_type:
            if t == "category":
                target.append(self.y[index])
            elif t == "annotation":
                data = scipy.io.loadmat(os.path.join(self.root,
                                                     "Annotations",
                                                     self.annotation_categories[self.y[index]],
                                                     "annotation_{:04d}.mat".format(self.index[index])))
                target.append(data["obj_contour"])
            else:
                raise ValueError("Target type \"{}\" is not recognized.".format(t))
        target = tuple(target) if len(target) > 1 else target[0]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _check_integrity(self):
        # can be more robust and check hash of files
        return os.path.exists(os.path.join(self.root, "101_ObjectCategories"))

    def __len__(self):
        return len(self.index)

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url("http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz",
                     self.root,
                     "101_ObjectCategories.tar.gz",
                     "b224c7392d521a49829488ab0f1120d9")
        download_url("http://www.vision.caltech.edu/Image_Datasets/Caltech101/Annotations.tar",
                     self.root,
                     "101_Annotations.tar",
                     "6f83eeb1f24d99cab4eb377263132c91")

        # extract file
        with tarfile.open(os.path.join(self.root, "101_ObjectCategories.tar.gz"), "r:gz") as tar:
            tar.extractall(path=self.root)

        with tarfile.open(os.path.join(self.root, "101_Annotations.tar"), "r:") as tar:
            tar.extractall(path=self.root)

    def extra_repr(self):
        return "Target type: {target_type}".format(**self.__dict__)


class Caltech256(VisionDataset):
    """`Caltech 256 <http://www.vision.caltech.edu/Image_Datasets/Caltech256/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``caltech256`` exists or will be saved to if download is set to True.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(self, root,
                 transform=None, target_transform=None,
                 download=False):
        super(Caltech256, self).__init__(os.path.join(root, 'caltech256'))
        os.makedirs(self.root, exist_ok=True)
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.categories = sorted(os.listdir(os.path.join(self.root, "256_ObjectCategories")))
        self.index = []
        self.y = []
        for (i, c) in enumerate(self.categories):
            n = len(os.listdir(os.path.join(self.root, "256_ObjectCategories", c)))
            self.index.extend(range(1, n + 1))
            self.y.extend(n * [i])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = Image.open(os.path.join(self.root,
                                      "256_ObjectCategories",
                                      self.categories[self.y[index]],
                                      "{:03d}_{:04d}.jpg".format(self.y[index] + 1, self.index[index])))

        target = self.y[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _check_integrity(self):
        # can be more robust and check hash of files
        return os.path.exists(os.path.join(self.root, "256_ObjectCategories"))

    def __len__(self):
        return len(self.index)

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url("http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar",
                     self.root,
                     "256_ObjectCategories.tar",
                     "67b4f42ca05d46448c6bb8ecd2220f6d")

        # extract file
        with tarfile.open(os.path.join(self.root, "256_ObjectCategories.tar"), "r:") as tar:
            tar.extractall(path=self.root)
