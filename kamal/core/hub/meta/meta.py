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