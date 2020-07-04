from copy import deepcopy
import os
import torch

class Pruner(object):
    def __init__(self, strategy):
        self.strategy = strategy
 
    def prune(self, model, rate=0.1, example_inputs=None):
        ori_num_params = sum( [ torch.numel(p) for p in model.parameters() ] )
        model = deepcopy(model).cpu()
        model = self._prune( model, rate=rate, example_inputs=example_inputs )
        new_num_params = sum( [ torch.numel(p) for p in model.parameters() ] )
        print( "%d=>%d, %.2f%% params were pruned"%( ori_num_params, new_num_params, 100*(ori_num_params-new_num_params)/ori_num_params ) )
        return model
    
    def _prune(self, model, **kargs):
        return self.strategy( model, **kargs)

        