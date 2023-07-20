import numpy as np
import torch
from kamal.core.metrics.stream_metrics import Metric
from typing import Callable
import ipdb

# 定义一个平均相对误差（rel）类，继承自度量（Metric）类
class MeanRelativeError(Metric):
    def __init__(self, attach_to=None):
        super(MeanRelativeError, self).__init__(attach_to=attach_to)
        self.reset()

    @torch.no_grad()
    def update(self, outputs, targets):
        outputs, targets = self._attach(outputs, targets)
        rel = torch.mean(torch.abs(outputs - targets))/torch.mean(targets)
        self._rel += torch.sum(rel)
        self._cnt += 1

    def get_results(self):
        return (self._rel / self._cnt).detach().cpu()
    
    def reset(self):
        self._rel = 0.0
        self._cnt = 0