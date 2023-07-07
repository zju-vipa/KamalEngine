import bisect
import os

import numpy as np
from numpy import random
import torchvision
import torchvision.datasets as datasets
from PIL import PngImagePlugin
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.cifar import CIFAR100
import torchvision.transforms as transforms
from tqdm import tqdm
from kamal.vision import sync_transforms as sT

LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)


def getDistributionInfo(dataset: Dataset):
    num_classes = len(dataset.classes)
    info = [0 for i in range(num_classes)]
    for i in tqdm(range(len(dataset))):
        _, label = dataset[i]
        info[label] += 1
    return info

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, target_transform=None) -> None:
        self.target_transform = target_transform
        self.dataset = dataset
        self.classes = dataset.classes
        self.idxs = idxs

    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, index: int):
        image, label = self.dataset[self.idxs[index]]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label

def getImageNet32(data_root='/home/by/ka/KamalEngine/examples/data/imagenet32/',with_index=True):
    train_trans=sT.Compose([
        sT.RandomCrop(32, padding=4),
        sT.RandomHorizontalFlip(),
        sT.RandomVerticalFlip(),
        sT.ToTensor(),
        sT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    trans=sT.Compose([
        sT.ToTensor(),
        sT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dir=os.path.join(data_root,'train_32x32/box')
    test_dir=os.path.join(data_root,'val')
    dataset_train=datasets.ImageFolder(train_dir,trans)
    dataset_train=DatasetWithDistribution(dataset_train)
    dataset_test=datasets.ImageFolder(test_dir,trans)
    dataset_test=DatasetWithDistribution(dataset_test,"/home/by/ka/KamalEngine/examples/knowledge_amalgamation/FedSA/cache/imagenet_test.npy")#记录数据分布情况
    if with_index:
        dataset_train=DatasetIndex(dataset_train)#通过 DatasetWithDistribution 函数将数据集对象（dataset_train 和 dataset_test）转换为带有分布信息的数据集对象。
        dataset_test=DatasetIndex(dataset_test)#使用 DatasetIndex 函数将数据集对象转换为带有索引信息的数据集对象。
    return dataset_train,dataset_test

class DatasetWithDistribution(Dataset):
    def __init__(self,set,distribution_file="/home/by/ka/KamalEngine/examples/knowledge_amalgamation/FedSA/cache/imagenet32.npy") -> None:
        super().__init__()
        self.set=set
        self.classes = set.classes
        self.class_to_idx = set.class_to_idx
        self.targets=set.targets
        if not os.path.exists(distribution_file):
            self.label_distribution=getDistributionInfo(set)
            np.save(distribution_file,np.array(self.label_distribution))
        else:
            self.label_distribution=np.load(distribution_file).tolist()

    def __len__(self):
        return len(self.set)
    
    def __getitem__(self, index: int):
        image, label = self.set[index]
        return image,label

class DatasetIndex(Dataset):
    def __init__(self, set1) -> None:
        super().__init__()
        self.set1 = set1
        self.classes = set1.classes
        self.class_to_idx = set1.class_to_idx
        

    def __len__(self) -> int:
        return len(self.set1)

    def __getitem__(self, index: int):
        image, label = self.set1[index]
        return index, image, label

def split_dataset(dataset, num_groups):
    """
    :param dataset:
    :param num_users:
    :return:
    """
    if isinstance(num_groups,int):
        assert len(dataset.classes)%num_groups==0
        dict_users = {i: np.array([], dtype='int64') for i in range(num_groups)}
        classes_groups=[len(dataset.classes)//num_groups]*num_groups
        n_split=num_groups
    else:
        assert sum(num_groups)==len(dataset.classes)
        dict_users = {i: np.array([], dtype='int64') for i in range(len(num_groups))}
        classes_groups=num_groups
        n_split=len(num_groups)
    
    idxs = np.arange(len(dataset))
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    current_pos=0
    array_idx=0
    # divide and assign
    start_idx=0
    for i in range(n_split):
        end_idx=start_idx+sum(dataset.label_distribution[current_pos:current_pos+classes_groups[array_idx]])
        dict_users[i]=idxs[start_idx:end_idx]
        start_idx=end_idx
        current_pos+=classes_groups[array_idx]
        array_idx+=1
    return dict_users

def getCIFAR100(data_root):
    CIFAR100_MEAN=(0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_STD=(0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    train_trans = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        # transforms.CenterCrop(24),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    ])
    test_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    ])
    dataset_train = datasets.CIFAR100(data_root,
                                      train=True,
                                      download=True,
                                      transform=train_trans)
    dataset_test = datasets.CIFAR100(data_root,
                                     train=False,
                                     download=True,
                                     transform=test_trans)
    return dataset_train, dataset_test

def get_sample_dataset(dataset,num_per_classes=100):
    n_classes=len(dataset.classes)
    counter=[0]*n_classes
    res_idxs=[]
    for i in range(random.randint(0,3000),len(dataset)):
        _,label=dataset[i]
        if(counter[label]<num_per_classes):
            res_idxs.append(i)
            counter[label]+=1
        temp=[1 if num==num_per_classes else 0 for num in counter]
        if(sum(temp)==n_classes):
            break
    return res_idxs

class MYDatasetConcat(Dataset):
    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self,train_datasets) -> None:
        train_datasets=list(train_datasets)
        self.classes=train_datasets[0].classes
        self.train_datasets=train_datasets
        self.num_datasets=len(train_datasets)
        self.cumulative_sizes=self.cumsum(train_datasets)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx: int):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        image,label = self.train_datasets[dataset_idx][sample_idx]
        return image,label