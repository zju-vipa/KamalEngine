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

    def __init__(self, datasets, tasks, transforms=None,
                 transform=None,
                 target_transform=None):
        super(LabelConcatDataset, self).__init__(root=None, transforms=transforms,
                                                 transform=transform, target_transform=target_transform)
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.tasks = tasks
        self._is_seg = ( 'Segmentation' in self.tasks )

    def __getitem__(self, idx):
        trans_list = []
        image = self.datasets[0].__getitem__(idx)[0]
        trans_list.append(image)
        if self.transforms is not None:
            for dataset in self.datasets:
                target = dataset.__getitem__(idx)[1]
                trans_list.append(target)
        outputs = self.transforms(*trans_list)
        if self._is_seg:
            index = self.tasks.index('Segmentation')
            outputs[index+1] = (outputs[index+1].to(dtype=torch.uint8) -1).to(dtype=torch.long)
        return outputs[0], [target.squeeze(0) for target in outputs[1:]]

    def __len__(self):
        return len(self.datasets[0].images)
