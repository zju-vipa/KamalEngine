import abc
from contextlib import contextmanager
from .. import metrics
from . import task
import torch
from tqdm import tqdm
from ...utils import set_mode

import sys

class EvaluatorBase(abc.ABC):
    def __init__(self, data_loader, task):
        self.data_loader = data_loader
        self.task = task

    @abc.abstractmethod
    def eval(self, model):
        pass

class ClassificationEvaluator(EvaluatorBase):
    def __init__(self, 
                data_loader, 
                task=task.ClassificationTask(),
                metric=None,
                progress=True):
        super(ClassificationEvaluator, self).__init__(data_loader, task)
        self.progress = progress
        if metric is None:
            metric = metrics.StreamClassificationMetrics()
        self.metric = metric
    
    def eval(self, model, device=None):
        device = device if device is not None else \
            torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        self.metric.reset()
        model.to(device)
        
        with torch.no_grad(), set_mode(model, training=False):
            for i, (inputs, targets) in enumerate( tqdm(self.data_loader, disable=not self.progress) ): 
                inputs, targets = inputs.to(device), targets.to(device)
                logits = self.task.get_logits( model, inputs )
                self.metric.update( logits, targets )
        return self.metric.get_results()

class SegmentationEvaluator(ClassificationEvaluator):
    def __init__(self, num_classes, data_loader, task=task.SegmentationTask(), progress=True):
        super( SegmentationEvaluator, self ).__init__(data_loader, task, progress)
        self.metric = metric.StreamSegmentationMetrics(num_classes, ignore_index=255)

class DepthEvaluator(EvaluatorBase):
    def __init__(self, data_loader, task=task.DepthTask(),progress=True):
        super(DepthEvaluator, self).__init__(data_loader, task)
        self.metric = metrics.StreamDepthMetrics(thresholds=[1.25, 1.25**2, 1.25**3])
        self.progress = progress
    
    def eval(self, model, device=None):
        device = device if device is not None else \
            torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        self.metric.reset()
        model.to(device)
        with torch.no_grad(), set_mode(model, training=False):
            for i, (images, targets) in enumerate( tqdm(self.data_loader, disable=not self.progress) ):
                images, targets = images.to(device), targets.to(device)
                outs = model( images ) 
                self.metric.update(outs, targets)

        return self.metric.get_results()

class CriterionEvaluator(EvaluatorBase):
    def __init__(self, data_loader, task, progress=True):
        super(CriterionEvaluator, self).__init__(data_loader, task)
        self.progress = progress

    def eval(self, model, device=None):
        device = device if device is not None else \
            torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        model.to(device)
        avg_loss = 0
        with torch.no_grad(), set_mode(model, training=False):
            for i, (inputs, targets) in enumerate( tqdm(self.data_loader, disable=not self.progress) ): 
                inputs, targets = inputs.to(device), targets.to(device)
                loss = self.task.get_loss( model, inputs, targets )['loss']
                avg_loss+=loss.item()
        return avg_loss/len(self.data_loader)

class MultitaskEvaluator(EvaluatorBase):
    def __init__(self, data_loader, split_size, tasks, task=task.SbmTask(), progress=True):
        super(MultitaskEvaluator, self).__init__(data_loader, task)
        self.metrics_list = []
        self.tasks = tasks
        for (teacher,task_name, num_classes) in zip(task.tasks, self.tasks, split_size):
            evaluator = getattr(sys.modules[__name__], task_name+'Evaluator')(num_classes, data_loader, teacher)
            self.metrics_list.append(evaluator.metrics)
        self.progress = progress

    def eval(self, model, split_size, device=None):
        results_dict = {}
        device = device if device is not None else \
            torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        for i in range(len(self.metrics_list)):
            self.metrics_list[i].reset()
        model.to(device)
        with torch.no_grad(), set_mode(model, training=False):
            for i, (images, targets_list) in enumerate( tqdm(self.data_loader, disable=not self.progress) ):
                images = images.to(device)
                joint_outputs = self.task.predict( model, images, split_size)['preds']
                for j, (targets, outs) in enumerate(zip(targets_list, joint_outputs)):
                    targets = targets.to(device)
                    if self.tasks[j] == 'Segmentation':
                         # multiple output
                        if isinstance(outs, (tuple, list)):
                            outs = outs[-1]
                        outs = outs.max(1)[1]
                    self.metrics_list[j].update(outs, targets)
        for task_name, metrics in zip(self.tasks, self.metrics_list):
            results_dict[task_name] = metrics.get_results()
        return results_dict

class SbmEvaluator(MultitaskEvaluator):
    def __init__(self, data_loader, split_size, tasks, task=task.SbmTask(), progress=True):
        super(SbmEvaluator, self).__init__(data_loader, split_size, tasks, task=task, progress=progress)

class KDClassificationEvaluator(EvaluatorBase):
    def __init__(self, 
                data_loader, 
                teacher,
                task=task.ClassificationTask(),
                progress=True):
        super(KDClassificationEvaluator, self).__init__(data_loader, task)
        self.metrics = metrics.StreamClassificationMetrics()
        self.progress = progress
        self.teacher = teacher

    def eval(self, model, device=None):
        device = device if device is not None else \
            torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        self.metrics.reset()
        teacher = self.teacher

        model.to(device)
        teacher.to(device)
        
        with torch.no_grad(), \
             set_mode(model, training=False), \
             set_mode( teacher, training=False ): 
            for i, data in enumerate( tqdm(self.data_loader, disable=not self.progress) ): 
                inputs = data[0] if isinstance( data, (tuple, list) ) else data
                inputs = inputs.to(device)
                s_preds = self.task.predict( model, inputs )['preds']
                t_preds = self.task.predict( teacher, inputs )['preds']
                self.metrics.update( s_preds, t_preds )
        return self.metrics.get_results()