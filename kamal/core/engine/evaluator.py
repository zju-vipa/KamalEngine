import abc
from contextlib import contextmanager
from .. import metrics
from . import task
import torch
from tqdm import tqdm
from .ctx import eval_ctx, device_ctx

class EvaluatorBase(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def eval(self, model):
        pass

class ClassificationEvaluator(EvaluatorBase):
    def __init__(self, data_loader):
        self.task = task.ClassificationTask()
        self.metrics = metrics.StreamClassificationMetrics()
        self.data_loader = data_loader
        
    def eval(self, model, device=None):
        device = device if device is not None else \
            torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        self.metrics.reset()
        with torch.no_grad(), eval_ctx(model), device_ctx( model, device ):
            for i, (inputs, targets) in enumerate( tqdm(self.data_loader) ): 
                inputs = inputs.to(device)
                preds = self.task.inference( model, inputs )['preds']
                self.metrics.update( preds, targets )
        return self.metrics.get_results()

class SegmentationEvaluator(ClassificationEvaluator):
    def __init__(self, num_classes, data_loader):
        super( SegmentationEvaluator, self ).__init__()
        self.metrics = metrics.StreamSegmentationMetrics(num_classes)
        self.task = task.SegmentationTask()
            
            