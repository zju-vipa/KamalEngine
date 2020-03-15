import torch
import torch.nn as nn 
import numpy as np
from copy import deepcopy
from hyperopt import fmin, tpe, hp
import pickle, os
import math
from ..core.engine.trainer import TrainerBase

class Pruner(object):
    def __init__(self, strategy):
        self.strategy = strategy
 
    def prune(self, model, rate=0.1, fake_input=None):
        ori_num_params = sum( [ torch.numel(p) for p in model.parameters() ] )
        model = deepcopy(model).cpu()
        self._prune( model, rate=rate, fake_input=fake_input )
        new_num_params = sum( [ torch.numel(p) for p in model.parameters() ] )
        print( "%d=>%d, %.2f%% params were pruned"%( ori_num_params, new_num_params, 100*(ori_num_params-new_num_params)/ori_num_params ) )
        return model
    
    def _prune(self, model, **kargs):
        return self.strategy( model, **kargs)

        