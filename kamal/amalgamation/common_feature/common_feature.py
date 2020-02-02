
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
from ...core.loss import CFLLoss
from .blocks import CFL_ConvBlock

class CommonFeatureLearning(nn.Module):
    """Common Feature Learning Algorithm

    Learn common features from multiple pretrained networks.
    https://arxiv.org/abs/1906.10546

    **Parameters:**
        - **layers** (list): student layer and teacher layers
        - **cfl_blocks** (list or nn.ModuleList): common feature blocks like ``CFL_ConvBlock``
    """

    def __init__(self, 
                 layers, 
                 num_features,
                 cfl_block=None,
                 sigmas=[0.001, 0.01, 0.05, 0.1, 0.2, 1, 2],
                 beta=1.0):
        super(CommonFeatureLearning, self).__init__()
        self.features = {}
        self.layers = layers

        def fetch_layer_output(module, input, output):
            self.features[module] = output
        s_layer, *t_layers = layers
        s_layer.register_forward_hook(fetch_layer_output)
        for t_layer in t_layers:
            t_layer.register_forward_hook(fetch_layer_output)

        if cfl_block is None:
            cfl_block = CFL_ConvBlock(num_features[0], num_features[1:], 128) 
        self.cfl_block = cfl_block
        self.cfl_criterion = CFLLoss( sigmas=sigmas, normalized=True, beta=beta)

    def forward(self, return_features=False):
        (s_layer, *t_layers) = self.layers
        s_feature = self.features[s_layer]
        t_features = [ self.features[t_layer].detach() for t_layer in t_layers ]
        
        (hs, hts), (fts_, fts) = self.cfl_block(s_feature, t_features)
        cfl_loss = self.cfl_criterion( hs, hts, fts_, fts ) 
        return cfl_loss
