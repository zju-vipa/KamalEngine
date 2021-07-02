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

        