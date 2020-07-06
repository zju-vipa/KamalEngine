import abc
import torch
import torch.nn as nn
import torch.nn.functional as F 
import sys
import typing

from typing import Callable, Dict
from kamal.core import criterions

from typing import Callable

class Criterion(nn.Module):
    def __init__(self, 
                 fn: Callable, 
                 scale: float=1.0, 
                 output_target_transform: Callable= lambda x, t: (x, t),
                ):
        super( Criterion, self ).__init__()
        self._fn = fn
        self._scale = scale
        self._output_target_transform = output_target_transform

    def forward(self, output, target):
        output, target = self._output_target_transform( output, target )
        loss = self._scale * self._fn( output, target )
        return loss

class Task(object):
    def __init__(self, criterion_dict: Dict):
        self._criterion_dict = criterion_dict

    def add_criterions(self, criterion_dict: Dict):
        self._criterion_dict.update(criterions)
    
    def get_loss( self, model, inputs, *targets ):
        if len(targets)==1:
            targets = targets[0]
        outputs = model(inputs)
        loss_dict = {}
        for key, cri in self._criterion_dict.items():
            if isinstance(cri, criterions.Criterion):
                loss_dict[key] = cri( outputs, targets )
        return loss_dict  

class StandardTask(object):
    @staticmethod
    def classification():
        return Task( criterion_dict={'CE': criterions.Criterion( F.cross_entropy )} )

    @staticmethod
    def regression():
        return Task( criterion_dict={'MSE': criterions.Criterion( F.mse_loss, output_target_transform=lambda x, y: (x.view_as(y), y) ) } )

    @staticmethod
    def segmentation():
        return Task( criterion_dict={'CE': criterions.Criterion( nn.CrossEntropyLoss(ignore_index=255) )} )
    
    @staticmethod
    def monocular_depth():
        return Task( criterion_dict={'L1': criterions.Criterion( F.l1_loss, output_target_transform=lambda x, y: (x.view_as(y), y) ) } )

    @staticmethod
    def detection():
        raise NotImplementedError

    @staticmethod
    def distillation():
        raise Task( criterions={'KLD': criterions.Criterion( criterions.functional.kldiv )} )