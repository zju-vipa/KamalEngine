from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from . import datasets as vipazoo_datasets


def get_cifar10_dataset(path, train, transform):
    sample_set = datasets.CIFAR10(
        path, train=train, transform=transform, download=True)
    return sample_set


def get_nyu_dataset(path, train, transform):
    sample_set = vipazoo_datasets.NYUDataset(
        path, train=train, transform=transform)
    return sample_set


datasets_map = {
    'cifar10': get_cifar10_dataset,
    'nyu': get_nyu_dataset
}

transforms_map = {
    'cifar10': transforms.Compose([
        transforms.Resize(128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    'nyu': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
}


def get_loader(name, path, train, batch_size, shuffle, num_workers=4):
    if name in datasets_map:
        sample_set = datasets_map[name](path, train, transforms_map[name])
        return DataLoader(sample_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    else:
        pass
