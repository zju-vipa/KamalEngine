from __future__ import division
import torch

import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Union, Any

class Metric(ABC):
    def __init__(self, output_target_transform: Callable=lambda x,y: (x,y) ):
        self._output_target_transform = output_target_transform

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


class MetricCompose(object):
    def __init__(self, metric_dict: dict, primary_metric: Union[str, Callable]):
        self._metric_dict = metric_dict
        if isinstance(primary_metric, str):
            assert primary_metric in self._metric_dict.keys()
        self._primary_metric = primary_metric
        
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

    def get_primary_metric(self, results: dict) -> (str, Any):
        if isinstance( self._primary_metric, str ):
            return self._primary_metric, results[self._primary_metric]
        elif isinstance(self._primary_metric, Callable):
            metric_name, metric_score = self._primary_metric( results )
            return metric_name, metric_score
        else:
            raise NotImplementedError

