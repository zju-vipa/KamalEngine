from torchvision.datasets import VisionDataset
from PIL import Image
import torch

class LabelConcatDataset(VisionDataset):
    """Dataset as a concatenation of dataset's lables.

    This class is useful to assemble the same dataset's labels.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
        tasks (list) : List of teacher tasks  
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry and returns a transformed version.
    """

    def __init__(self, datasets, transforms=None, transform=None, target_transform=None):
        super(LabelConcatDataset, self).__init__(
            root=None, transforms=transforms, transform=transform, target_transform=target_transform)
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)

    def __getitem__(self, idx):
        targets_list = []
        for dst in self.datasets:
            image, target = dst[idx]
            targets_list.append(target)
        if self.transforms is not None:
            image, *targets_list = self.transforms( image, *targets_list ) 
        return image, [*targets_list]

    def __len__(self):
        return len(self.datasets[0].images)
