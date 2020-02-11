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
                task=task.ClassificationTask()):
        super(ClassificationEvaluator, self).__init__(data_loader, task)
        self.metrics = metrics.StreamClassificationMetrics()
    
    def eval(self, model, device=None):
        device = device if device is not None else \
            torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        self.metrics.reset()
        model.to(device)
        
        with torch.no_grad(), set_mode(model, training=False):
            for i, (inputs, targets) in enumerate( tqdm(self.data_loader) ): 
                inputs = inputs.to(device)
                preds = self.task.inference( model, inputs )['preds']
                self.metrics.update( preds, targets )
        return self.metrics.get_results()

class SegmentationEvaluator(ClassificationEvaluator):
    def __init__(self, num_classes, data_loader, task=task.SegmentationTask()):
        super( SegmentationEvaluator, self ).__init__(data_loader, task)
        self.metrics = metrics.StreamSegmentationMetrics(num_classes, ignore_index=255)

class DepthEvaluator(EvaluatorBase):
    def __init__(self, data_loader, task=task.DepthTask()):
        super(DepthEvaluator, self).__init__(data_loader, task)
        self.metrics = metrics.StreamDepthMetrics(thresholds=[1.25, 1.25**2, 1.25**3])

class CriterionEvaluator(EvaluatorBase):
    def __init__(self, data_loader, task):
        super(CriterionEvaluator, self).__init__(data_loader, task)

    def eval(self, model, device=None):
        device = device if device is not None else \
            torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        model.to(device)
        avg_loss = 0
        with torch.no_grad(), set_mode(model, training=False):
            for i, (inputs, targets) in enumerate( tqdm(self.data_loader) ): 
                inputs, targets = inputs.to(device), targets.to(device)
                loss = self.task.get_loss( model, inputs, targets )['loss']
                avg_loss+=loss.item()
        return avg_loss/len(self.data_loader)