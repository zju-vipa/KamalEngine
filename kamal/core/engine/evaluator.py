import abc
from contextlib import contextmanager
from .. import metrics
from . import task
import torch
from tqdm import tqdm
from .trainer import set_mode

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
                progress=True):
        super(ClassificationEvaluator, self).__init__(data_loader, task)
        self.metrics = metrics.StreamClassificationMetrics()
        self.progress = progress

    def eval(self, model, device=None):
        device = device if device is not None else \
            torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        self.metrics.reset()
        model.to(device)
        
        with torch.no_grad(), set_mode(model, training=False):
            for i, (inputs, targets) in enumerate( tqdm(self.data_loader, disable=not self.progress) ): 
                inputs = inputs.to(device)
                preds = self.task.predict( model, inputs )['preds']
                self.metrics.update( preds, targets )
        return self.metrics.get_results()

class SegmentationEvaluator(ClassificationEvaluator):
    def __init__(self, num_classes, data_loader, task=task.SegmentationTask(), progress=True):
        super( SegmentationEvaluator, self ).__init__(data_loader, task, progress)
        self.metrics = metrics.StreamSegmentationMetrics(num_classes, ignore_index=255)

class DepthEvaluator(EvaluatorBase):
    def __init__(self, data_loader, task=task.DepthTask(),progress=True):
        super(DepthEvaluator, self).__init__(data_loader, task)
        self.metrics = metrics.StreamDepthMetrics(thresholds=[1.25, 1.25**2, 1.25**3])
        self.progress = progress
    
    def eval(self, model, device=None):
        device = device if device is not None else \
            torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        self.metrics.reset()
        model.to(device)
        with torch.no_grad(), set_mode(model, training=False):
            for i, (images, targets) in enumerate( tqdm(self.data_loader, disable=not self.progress) ):
                images, targets = images.to(device), targets.to(device)
                outs = model( images ) 
                self.metrics.update(outs, targets)

        return self.metrics.get_results()

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