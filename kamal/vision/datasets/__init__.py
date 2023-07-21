from .ade20k import ADE20K
from .caltech import Caltech101, Caltech256
from .camvid import CamVid
from .cityscapes import Cityscapes
from .cub200 import CUB200
from .fgvc_aircraft import FGVCAircraft
from .nyu import NYUv2
from .stanford_cars import StanfordCars
from .stanford_dogs import StanfordDogs
from .sunrgbd import SunRGBD
from .voc import VOCClassification, VOCSegmentation
from .dataset import LabelConcatDataset
from .fskd_cifarfew import *
from .CelebA import *
from .customize_class_data import *
from .taskonomy import Taskonomy

from torchvision import datasets as torchvision_datasets

from .unlabeled import UnlabeledDataset
from .preprocess import build_dataset

from torchvision import datasets as torchvision_datasets

from .unlabeled import UnlabeledDataset
# from .preprocess import build_dataset

from torch.utils.data import Dataset
from .taskonomy import Taskonomy

from torchvision import datasets as torchvision_datasets

from .unlabeled import UnlabeledDataset

from .cifar import get_cifar_10, get_cifar_100, get_cifar_10_
from .imagenet import get_imagenet
from .tiny_imagenet import get_tiny_imagenet


DATASET_DICT = {
    "cifar-10": get_cifar_10_,
    "cifar-100": get_cifar_100,
    "cifar10": get_cifar_10_,
    "cifar100": get_cifar_100,
    "imagenet": get_imagenet,
    "tiny-imagenet": get_tiny_imagenet
}


def get_dataset(name: str, root: str, split: str = "train", **kwargs) -> Dataset:
    fn = DATASET_DICT[name]
    return fn(root=root, split=split)
