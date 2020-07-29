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