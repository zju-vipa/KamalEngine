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

import torch_pruning as tp
import abc
import torch
import torch.nn as nn
import random
import numpy as np 

_PRUNABLE_MODULES= tp.DependencyGraph.PRUNABLE_MODULES

class BaseStrategy(abc.ABC):

    @abc.abstractmethod
    def select(self, layer_to_prune):
        pass

    def  __call__(self, model, rate=0.1, example_inputs=None):
        if example_inputs is None:
            example_inputs = torch.randn( 1,3,256,256 )

        DG = tp.DependencyGraph()
        DG.build_dependency(model, example_inputs=example_inputs)

        prunable_layers = []
        total_params = 0
        num_accumulative_conv_params = [ 0, ]

        for m in model.modules():
            if isinstance(m, _PRUNABLE_MODULES ) :
                nparam = tp.utils.count_prunable_params( m )
                total_params += nparam
                if isinstance(m, (nn.modules.conv._ConvNd, nn.Linear)):
                    prunable_layers.append( m )
                    num_accumulative_conv_params.append( num_accumulative_conv_params[-1]+nparam )
        prunable_layers.pop(-1) # remove the last layer
        num_accumulative_conv_params.pop(-1) # remove the last layer
        
        num_conv_params = num_accumulative_conv_params[-1]
        num_accumulative_conv_params = [ ( num_accumulative_conv_params[i], num_accumulative_conv_params[i+1] ) for i in range(len(num_accumulative_conv_params)-1) ]
        
        def map_param_idx_to_conv_layer(i):
            for l, accu in zip( prunable_layers, num_accumulative_conv_params ):
                if accu[0]<=i and i<accu[1]:
                    return l 

        num_pruned = 0
        while num_pruned<total_params*rate:
            layer_to_prune = map_param_idx_to_conv_layer( random.randint( 0, num_conv_params-1 ) )
            if layer_to_prune.weight.shape[0]<1:
                continue
            idx = self.select( layer_to_prune )
            fn = tp.prune_conv if isinstance(layer_to_prune, nn.modules.conv._ConvNd) else tp.prune_linear
            plan = DG.get_pruning_plan( layer_to_prune, fn, idxs=idx ) 
            num_pruned += plan.exec() 
        return model

class RandomStrategy(BaseStrategy):
    def select(self, layer_to_prune):
        return [ random.randint( 0, layer_to_prune.weight.shape[0]-1 ) ]

class LNStrategy(BaseStrategy):
    def __init__(self, n=2):
        self.n = n

    def select(self, layer_to_prune):
        w = torch.flatten( layer_to_prune.weight, 1 )
        norm = torch.norm(w, p=self.n, dim=1)
        idx = [ int(norm.min(dim=0)[1].item()) ]
        return idx