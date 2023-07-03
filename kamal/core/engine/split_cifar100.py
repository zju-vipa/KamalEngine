import os
import pickle
import sys

import numpy as np
from PIL import Image

from torch.utils.data import Dataset


def load_split_cifar100(root):
    data_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"]
    ]

    base_folder = 'cifar-100-python'

    for file_name, _ in data_list:
        data = []
        targets = []
        file_path = os.path.join(root, base_folder, file_name)
        with open(file_path, 'rb') as f:
            if sys.version_info[0] == 2:
                entry = pickle.load(f)
            else:
                entry = pickle.load(f, encoding='latin1')
            data.append(entry['data'])
            if 'fine_labels' in entry:
                targets.extend(entry['fine_labels'])
            else:
                targets.extend(entry['coarse_labels'])
        
        data = np.vstack(data)
        
        half_split_cifar100(root, file_name, data, targets)


def half_split_cifar100(root, filename, data, targets):
    dataset_part1 = [[],[]]
    dataset_part2 = [[],[]]
    
    for i, target in enumerate(targets):
        if target < 50:
            dataset_part1[0].append(data[i])
            dataset_part1[1].append(target)
        else:
            dataset_part2[0].append(data[i])
            dataset_part2[1].append(target-50)
            
    dataset_part1[0] = np.vstack(dataset_part1[0]).reshape(-1, 3, 32, 32)
    dataset_part2[0] = np.vstack(dataset_part2[0]).reshape(-1, 3, 32, 32)
    
    with open(os.path.join(root, 'cifar100-{}-part0.pkl'.format(filename)), 'wb') as f:
        pickle.dump(dataset_part1, f)
    
    with open(os.path.join(root, 'cifar100-{}-part1.pkl'.format(filename)), 'wb') as f:
        pickle.dump(dataset_part2, f)
    
    
class CIFAR100_PART(Dataset):
    def __init__(self, root, train, part, transform=None):
        self.train = train
        self.transform = transform

        with open(os.path.join(root, 'cifar100-{}-part{}.pkl'.format("train" if self.train else "test", part)), 'rb') as f:
            entry = pickle.load(f)

        self.data, self.label = entry

        self.data = self.data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):
        img = self.data[index]
        label = self.label[index]
                  
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    root = '/home/jovyan/Python/MosaicKD/data/torchdata'
    load_split_cifar100(root)
    c100_train1 = CIFAR100_PART(root, train=True, part = 0)
    c100_train2 = CIFAR100_PART(root, train=True, part = 1)
    c100_test1 = CIFAR100_PART(root, train=False, part = 0)
    c100_test2 = CIFAR100_PART(root, train=False, part = 1)
    # print(c100_train1[110])
    # print(c100_train2[110])
    # print(c100_test1[110])
    # print(c100_test2[1])