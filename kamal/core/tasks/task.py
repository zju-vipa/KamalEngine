import abc
import torch
import torch.nn as nn
import torch.nn.functional as F 
import sys
import typing
from typing import Callable, Dict, List
from collections import Mapping, Sequence
from . import loss
from kamal.core import metrics, exceptions
from kamal.core.attach import AttachTo


class Task(object):
    def __init__(self, 
                 name: str, 
                 loss_fn: Callable,
                 scaling:float=1.0,
                 pred_fn: Callable=lambda x: x,
                 forward_fn: Callable=lambda model, x: model(x),
                 attach_to=None):
        self._attach = AttachTo(attach_to)
        self.loss_fn = loss_fn
        self.pred_fn = pred_fn
        self.forward_fn = forward_fn
        self.scaling = scaling
        self.name = name

    def forward(self, model, input):
        return self.forward_fn( model, input )
    
    def get_loss(self, outputs, targets):
        outputs, targets = self._attach(outputs, targets)
        return { self.name: self.loss_fn( outputs, targets ) * self.scaling }

    def predict(self, outputs):
        outputs = self._attach(outputs)
        return self.pred_fn(outputs)
        
    def __repr__(self):
        rep = "Task: [%s loss_fn=%s scaling=%.4f attach=%s]"%(self.name, str(self.loss_fn), self.scaling, self._attach)
        return rep


class Task(object):
    def __init__(self, 
                 name: str, 
                 loss_fn: Callable, 
                 scaling:float=1.0, 
                 pred_fn: Callable=lambda x: x,
                 attach_to=None):
        self._attach = AttachTo(attach_to)
        self.loss_fn = loss_fn
        self.pred_fn = pred_fn
        self.scaling = scaling
        self.name = name

    def get_loss(self, outputs, targets):
        outputs, targets = self._attach(outputs, targets)
        return { self.name: self.loss_fn( outputs, targets ) * self.scaling }

    def predict(self, outputs):
        outputs = self._attach(outputs)
        return self.pred_fn(outputs)
        
    def __repr__(self):
        rep = "Task: [%s loss_fn=%s scaling=%.4f attach=%s]"%(self.name, str(self.loss_fn), self.scaling, self._attach)
        return rep

class TaskCompose(list):
    def __init__(self, tasks:list):
        for task in tasks:
            if isinstance(task, Task):
                self.append(task)
    
    def get_loss(self, outputs, targets):
        loss_dict = {}
        for task in self:
            loss_dict.update( task.get_loss( outputs, targets ) )
        return loss_dict
    
    def predict(self, outputs):
        results = []
        for task in self:
            results.append( task.predict( outputs ) )
        return results
        
    def __repr__(self):
        rep="TaskCompose: \n"
        for task in self:
            rep+="\t%s\n"%task

class StandardTask:
    @staticmethod
    def classification(name='ce', scaling=1.0, attach_to=None):
        return Task( name=name, 
                     loss_fn=nn.CrossEntropyLoss(), 
                     scaling=scaling, 
                     pred_fn=lambda x: x.max(1)[1], 
                     attach_to=attach_to )

    @staticmethod
    def binary_classification(name='bce', scaling=1.0, attach_to=None):
        return Task(name=name, 
                    loss_fn=F.binary_cross_entropy_with_logits, 
                    scaling=scaling, 
                    pred_fn=lambda x: (x>0.5), 
                    attach_to=attach_to )

    @staticmethod
    def regression(name='mse', scaling=1.0, attach_to=None):
        return Task(name=name, 
                    loss_fn=nn.MSELoss(), 
                    scaling=scaling,
                    pred_fn=lambda x: x, 
                    attach_to=attach_to  )

    @staticmethod
    def segmentation(name='ce', scaling=1.0, attach_to=None):
        return Task(name=name, 
                    loss_fn=nn.CrossEntropyLoss(ignore_index=255), 
                    scaling=scaling, 
                    pred_fn=lambda x: x.max(1)[1], 
                    attach_to=attach_to )
    
    @staticmethod
    def monocular_depth(name='l1', scaling=1.0, attach_to=None):
        return Task(name=name, 
                    loss_fn=nn.L1Loss(), 
                    scaling=scaling, 
                    pred_fn=lambda x: x, 
                    attach_to=attach_to)

    @staticmethod
    def detection():
        raise NotImplementedError

    @staticmethod
    def distillation(name='kld', T=1.0, scaling=1.0, attach_to=None):
        return Task(name=name, 
                    loss_fn=loss.KLDiv(T=T), 
                    scaling=scaling, 
                    pred_fn=lambda x: x.max(1)[1],
                    attach_to=attach_to)


class StandardMetrics(object):

    @staticmethod
    def classification(attach_to=None):
        return metrics.MetricCompose(
            metric_dict={'acc': metrics.Accuracy(attach_to=attach_to)}
        )

    @staticmethod
    def regression(attach_to=None):
        return metrics.MetricCompose(
            metric_dict={'mse': metrics.MeanSquaredError(attach_to=attach_to)}
        )

    @staticmethod
    def segmentation(num_classes, ignore_idx=255, attach_to=None):
        confusion_matrix = metrics.ConfusionMatrix(num_classes=num_classes, ignore_idx=ignore_idx, attach_to=attach_to)
        return metrics.MetricCompose(
            metric_dict={'acc': metrics.Accuracy(attach_to=attach_to), 
                         'confusion_matrix': confusion_matrix , 
                         'miou': metrics.mIoU(confusion_matrix)}
        )

    @staticmethod
    def monocular_depth(attach_to=None):
        return metrics.MetricCompose(
            metric_dict={
                'rmse': metrics.RootMeanSquaredError(attach_to=attach_to),
                'rmse_log': metrics.RootMeanSquaredError( log_scale=True,attach_to=attach_to ),
                'rmse_scale_inv': metrics.ScaleInveriantMeanSquaredError(attach_to=attach_to),
                'abs rel': metrics.AbsoluteRelativeDifference(attach_to=attach_to),
                'sq rel': metrics.SquaredRelativeDifference(attach_to=attach_to), 
                'percents within thresholds': metrics.Threshold( thresholds=[1.25, 1.25**2, 1.25**3], attach_to=attach_to )
            }
        )
        
    @staticmethod
    def loss_metric(loss_fn):
        return metrics.MetricCompose(
            metric_dict={
                'loss': metrics.AverageMetric( loss_fn )
            }
        )
