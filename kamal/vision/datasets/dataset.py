# Copyright 2020 Zhejiang Lab. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================


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
        for d in self.datasets:
            for t in ('transform', 'transforms', 'target_transform'):
                if hasattr( d, t ):
                    setattr( d, t, None )

    def __getitem__(self, idx):
        targets_list = []
        for dst in self.datasets:
            image, target = dst[idx]
            targets_list.append(target)
        if self.transforms is not None:
            image, *targets_list = self.transforms( image, *targets_list ) 
        data = [ image, *targets_list ]
        return data

    def __len__(self):
        return len(self.datasets[0].images)
