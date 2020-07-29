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

from __future__ import division
import torch

import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Union, Any, Mapping, Sequence
import numbers
from kamal.core.attach import AttachTo

class Metric(ABC):
    def __init__(self, attach_to=None):
        self._attach = AttachTo(attach_to)

    @abstractmethod
    def update(self, pred, target):
        """ Overridden by subclasses """
        raise NotImplementedError()
    
    @abstractmethod
    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()


class MetricCompose(dict):
    def __init__(self, metric_dict: Mapping):
        self._metric_dict = metric_dict

    def add_metrics( self, metric_dict: Mapping):
        if isinstance(metric_dict, MetricCompose):
            metric_dict = metric_dict.metrics
        self._metric_dict.update(metric_dict)
        return self

    @property
    def metrics(self):
        return self._metric_dict
        
    @torch.no_grad()
    def update(self, outputs, targets):
        for key, metric in self._metric_dict.items():
            if isinstance(metric, Metric):
                metric.update(outputs, targets)
    
    def get_results(self):
        results = {}
        for key, metric in self._metric_dict.items():
            if isinstance(metric, Metric):
                results[key] = metric.get_results()
        return results

    def reset(self):
        for key, metric in self._metric_dict.items():
            if isinstance(metric, Metric):
                metric.reset()

    def __getitem__(self, name):
        return self._metric_dict[name]


