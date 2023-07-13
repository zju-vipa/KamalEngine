import abc
import torch
import torch.nn as nn
import torch.nn.functional as F 
import sys
import typing
from typing import Callable, Dict, List, Any
from collections import Mapping, Sequence
from . import loss
from kamal.core import metrics, exceptions
from kamal.core.attach import AttachTo

class C2KDTask(Task):
    def __init__(self, 
                 name: str, 
                 ce_criterion: Callable, 
                 DSA_criterion: Callable,
                 ESA_criterion: Callable,
                 error_calculator=error_calculator,
                 scaling:float=1.0, 
                 pred_fn: Callable=lambda x: x,
                 attach_to=None):
        super(C2KD_task, self).__init__(name)
        self._attach = AttachTo(attach_to)
        self.ce_criterion = ce_criterion
        self.DSA_criterion = DSA_criterion
        self.ESA_criterion = ESA_criterion
        self.pred_fn = pred_fn
        self.scaling = scaling

    def __repr__(self):
        rep = "Task: [%s loss_fn=%s scaling=%.4f attach=%s]"%(self.name, str(self.loss_fn), self.scaling, self._attach)
        return rep