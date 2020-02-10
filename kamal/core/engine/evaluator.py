import abc
from contextlib import contextmanager
from .. import metrics
from . import task
import torch
from tqdm import tqdm
from .trainer import set_mode

class EvaluatorBase(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def eval(self, model):
        pass

class ClassificationEvaluator(EvaluatorBase):
    def __init__(self, data_loader, task=task.ClassificationTask()):
        self.task = task
        self.metrics = metrics.StreamClassificationMetrics()
        self.data_loader = data_loader
        
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
        self.metrics = metrics.StreamSegmentationMetrics(num_classes)