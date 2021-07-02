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

from copy import deepcopy
import abc, os

from . import TASK
import torch

__all__ = ['yaml', 'ImageInput', 'MetaData', 'AtlasEntryBase']

# Model Metadata
def Metadata( name:           str,
              dataset:        str,
              task:           int,
              url:            str,
              input:          dict,
              entry_args:     dict,
              other_metadata: dict):
    if task in [TASK.SEGMENTATION, TASK.CLASSIFICATION]:
        assert 'num_classes' in other_metadata
    return dict(
            name=name, 
            dataset=dataset, 
            task=task,
            url=url, 
            input=dict(input), 
            entry_args=entry_args,
            other_metadata=other_metadata
        )