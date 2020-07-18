from copy import deepcopy
import abc, os

from . import TASK
import torch

__all__ = ['yaml', 'ImageInput', 'MetaData', 'AtlasEntryBase']

# Model Metadata
def Metadata( name:           str=None,
              dataset:        str=None,
              task:           int=None,
              url:            str=None,
              input:          dict=None,
              entry_args:     dict=None,
              other_metadata: dict=None):
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