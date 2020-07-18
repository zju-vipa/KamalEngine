import numpy as np
import torch
from kamal.core.metrics.stream_metrics import Metric
from typing import Callable

__all__=['AverageMetric']

class AverageMetric(Metric):
    def __init__(self, fn:Callable, attach_to=None):
        super(AverageMetric, self).__init__(attach_to=attach_to)
        self._fn = fn
        self.reset()

    @torch.no_grad()
    def update(self, outputs, targets):

        outputs, targets = self._attach(outputs, targets)
        m = self._fn( outputs, targets )

        if m.ndim > 1:
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