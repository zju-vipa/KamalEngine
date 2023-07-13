from torchvision import datasets, transforms as T

from PIL import PngImagePlugin

from kamal.vision.models.classification import wresnet, resnet, vgg

LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024 ** 2)

import os

NORMALIZE_DICT = {
    # In-domain data
    'cifar100': dict(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    # Out-of-domain data
    'cifar10': dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    'imagenet': dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    'svhn': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'places365_32x32': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'imagenet_32x32': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
}

MODEL_DICT = {
    'wrn16_1': wresnet.wrn_16_1,
    'wrn16_2': wresnet.wrn_16_2,
    'wrn40_1': wresnet.wrn_40_1,
    'wrn40_2': wresnet.wrn_40_2,
    'resnet18': resnet.resnet18,
    'resnet34': resnet.resnet34,
    'vgg11': vgg.vgg11_bn,
    # 'resnet8': classifiers.resnet_tiny.resnet8,
    # 'resnet20': classifiers.resnet_tiny.resnet20,
    # 'resnet32': classifiers.resnet_tiny.resnet32,
    # 'resnet56': classifiers.resnet_tiny.resnet56,
    # 'resnet110': classifiers.resnet_tiny.resnet110,
    # 'resnet8x4': classifiers.resnet_tiny.resnet8x4,
    # 'resnet32x4': classifiers.resnet_tiny.resnet32x4,
    # 'resnet50':  classifiers.resnet.resnet50,
    # 'vgg8': classifiers.vgg.vgg8_bn,
    # 'vgg13': classifiers.vgg.vgg13_bn,
    # 'shufflenetv2': classifiers.shufflenetv2.shuffle_v2,
    # 'mobilenetv2': classifiers.mobilenetv2.mobilenet_v2,
}


def get_model(name: str, num_classes, pretrained=False, **kwargs):
    model = MODEL_DICT[name](num_classes=num_classes)
    return model


def get_dataset(name: str, data_root: str = 'data', return_transform=False):
    name = name.lower()
    data_root = os.path.expanduser(data_root)
    if name == 'cifar10':
        num_classes = 10
        train_transform = T.Compose([
            # T.Resize((224, 224), Image.BICUBIC),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        val_transform = T.Compose([
            # T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        data_root = os.path.join(data_root, 'torchdata')
        train_dst = datasets.CIFAR10(data_root, train=True, download=True, transform=train_transform)
        val_dst = datasets.CIFAR10(data_root, train=False, download=True, transform=val_transform)
    elif name == 'cifar100':
        num_classes = 100
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        data_root = os.path.join(data_root, 'torchdata')
        train_dst = datasets.CIFAR100(data_root, train=True, download=True, transform=train_transform)
        val_dst = datasets.CIFAR100(data_root, train=False, download=True, transform=val_transform)
    elif name == 'svhn':
        num_classes = 10
        train_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        data_root = os.path.join(data_root, 'torchdata')
        train_dst = datasets.SVHN(data_root, split='train', download=True, transform=train_transform)
        val_dst = datasets.SVHN(data_root, split='test', download=True, transform=val_transform)
    elif name == 'imagenet_32x32':
        num_classes = 1000
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        data_root = os.path.join(data_root, 'ImageNet_32x32')
        train_dst = datasets.ImageFolder(os.path.join(data_root, 'train'), transform=train_transform)
        val_dst = datasets.ImageFolder(os.path.join(data_root, 'val'), transform=val_transform)
    elif name == 'places365_32x32':
        num_classes = 365
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        data_root = os.path.join(data_root, 'Places365_32x32')
        train_dst = datasets.ImageFolder(os.path.join(data_root, 'train'), transform=train_transform)
        val_dst = datasets.ImageFolder(os.path.join(data_root, 'val'), transform=val_transform)
    else:
        raise NotImplementedError

    if return_transform:
        return num_classes, train_dst, val_dst, train_transform, val_transform
    return num_classes, train_dst, val_dst
