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
from torchvision import datasets as torchvision_datasets

from .unlabeled import UnlabeledDataset
from .preprocess import build_dataset