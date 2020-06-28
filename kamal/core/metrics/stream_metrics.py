from __future__ import division
import torch

import numpy as np
from abc import ABC, abstractmethod

class StreamMetricsBase(ABC):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def __call__(self, pred, target):
        self.update(pred, target)

    @abstractmethod
    def update(self, pred, target):
        """ Overridden by subclasses """
        raise NotImplementedError()
    
    @abstractmethod
    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    @abstractmethod
    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()


class AverageMeter(object):
    """ Average Record
    """
    def __init__(self):
        self.record = dict()

    def update(self, **kargs):
        for k, v in kargs.items():
            rec = self.record.get(k, None)
            if rec is None:
                self.record[k] = {'val': v, 'count': 1}  # init
            else:
                rec['val'] += v
                rec['count'] += 1

    def get_results(self, *keys):
        if len(keys)==0:
            keys = self.record.keys()
        return {k: (self.record[k]['val']/self.record[k]['count']) for k in args}
    
    def reset(self, *keys):
        if len(keys)==0:
            self.record = dict()
            return
        
        for k in keys:
            self.record[k] = {'val': 0.0, 'count': 0}
