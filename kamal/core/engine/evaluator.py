import abc, sys
import torch
from tqdm import tqdm

from kamal.core import metrics
from kamal.utils import set_mode

from typing import Union, Callable

class EvaluatorBase(abc.ABC):
    
    def __init__(self, metric):
        self._metric = metric

    @property
    def metric(self):
        return self._metric

    @abc.abstractmethod
    def eval(self, model):
        pass

class BasicEvaluator(EvaluatorBase):
    def __init__(self,
                 data_loader: torch.utils.data.DataLoader,
                 metric: metrics.MetricCompose,
                 progress: bool=False ):
        super( BasicEvaluator, self ).__init__(metric)
        self._data_loader = data_loader
        self.progress = progress
        
    def eval(self, model, device=None):
        device = device if device is not None else \
            torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        model.to(device)
        self._metric.reset()
        with torch.no_grad(), set_mode(model, training=False):
            for i, data in enumerate( tqdm(self._data_loader, disable=not self.progress) ):
                inputs, *targets = [ d.to(device) for d in data ]
                if len(targets)==1:
                    targets = targets[0]
                outputs = model( inputs )
                self._metric.update( outputs, targets )
        return self._metric.get_results()
