import torch_pruning
import abc
import torch
import torch.nn as nn
import random
import numpy as np 

_PRUNABLE_LAYER= ( nn.modules.conv._ConvNd, nn.modules.batchnorm._BatchNorm, nn.PReLU, nn.Linear )

class BaseStrategy(abc.ABC):
    @abc.abstractmethod
    def __call__(self, model):
        pass


class RandomStrategy(BaseStrategy):
    def  __call__(self, model, rate=0.1, fake_input=None):
        if fake_input is None:
            fake_input = torch.randn( 1,3,256,256 )
        DG = torch_pruning.DependencyGraph(model, fake_input=fake_input)
        
        conv_layers = []
        total_params = 0
        num_accumulative_conv_params = [ 0, ]

        for m in model.modules():
            if isinstance(m, _PRUNABLE_LAYER ) :
                nparam = torch_pruning.utils.count_prunable_params( m )  # number of conv kernels
                total_params += nparam
                if isinstance(m, nn.modules.conv._ConvNd):
                    conv_layers.append( m )
                    num_accumulative_conv_params.append( num_accumulative_conv_params[-1]+nparam )

        num_conv_params = num_accumulative_conv_params[-1]
        num_accumulative_conv_params = [ ( num_accumulative_conv_params[i], num_accumulative_conv_params[i+1] ) for i in range(len(num_accumulative_conv_params)-1) ]

        def map_param_idx_to_conv_layer(i):
            for l, accu in zip( conv_layers, num_accumulative_conv_params ):
                if accu[0]<i and i<accu[1]:
                    return l 

        num_pruned = 0
        while num_pruned<total_params*rate:
            layer_to_prune = map_param_idx_to_conv_layer( random.randint( 0, num_conv_params-1 ) )
            if layer_to_prune.weight.shape[0]<1:
                continue
            idx = [ random.randint( 0, layer_to_prune.weight.shape[0]-1 ) ]
            plan = DG.get_pruning_plan( layer_to_prune, torch_pruning.prune_conv, idxs=idx )        
            num_pruned += plan.exec() 
        return model