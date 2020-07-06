import numpy as np
import torch
from kamal.core.metrics.stream_metrics import Metric
from typing import Callable

__all__=['AverageMetric']

class AverageMetric(Metric):
    def __init__(self, fn:Callable, output_target_transform: Callable=lambda x,y: (x,y)):
        super(AverageMetric, self).__init__(output_target_transform=output_target_transform)
        self._fn = fn
        self.reset()

    @torch.no_grad()
    def update(self, outputs, targets):
        outputs, targets = self._output_target_transform(outputs, targets)
        m = self._fn( outputs, targets )
        if x.ndim > 1:
            self._cnt += m.shape[0]
            self._accum += m.sum(0)
        else:
            self._cnt += 1
            self._accum += m
    
    def get_results(self):
        return (self._accum / self._cnt).detach().cpu()
    
    def reset(self):
        self._accum = 0.
        self._cnt = 0.