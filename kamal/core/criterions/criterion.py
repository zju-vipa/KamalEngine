from typing import Callable
import torch.nn as nn

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