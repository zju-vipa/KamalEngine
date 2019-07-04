
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch

from ..utils import get_layer


class AmalNet(nn.Module):
    """ Network wrapper for Knowledge Kamalgamation
    It is used to extract intermediate features of networks for CFL or other KA algorithmns.
    """
    def __init__(self, module):
        super(AmalNet, self).__init__()
        self.module = module
        self.endpoints = list()  # init empty
        self.endpoints_info = list()

    def forward(self, x):
        self.endpoints = list()
        out = self.module(x)
        return out

    def register_endpoints(self, parse_fn=None, layer_parser=None, input_channel=3):
        """ Register Endpoints for specified layers in network.
        The registered features will be saved as endpoints.
        """
        self.endpoints_info = []

        # use parse_fn
        if parse_fn is not None:
            for (layer, info) in parse_fn(self.module):
                self._register(layer, info)
        # all layers
        else:
            assert layer_parser is not None, "layer_parser should not be None"
            for idx, layer in enumerate(get_layer(self.module, only_leaf=False)):
                info = layer_parser.parse(layer)
                if info is not None:  # default: all layers with parameters
                    self._register(layer, info)

    def _register(self, layer, info):
        layer.register_forward_hook(self._obtain_intermediate)
        self.endpoints_info.append(info)

    def _obtain_intermediate(self, module, input, output):
        self.endpoints.append(output)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.module.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict)
