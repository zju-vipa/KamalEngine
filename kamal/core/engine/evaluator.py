import abc, sys
import torch
from tqdm import tqdm
from kamal.core import metrics
from kamal.utils import set_mode
from typing import Any, Callable
from .engine import Engine
from .events import DefaultEvents
from kamal.core import callbacks

import weakref
from kamal.utils import move_to_device, split_batch

class BasicEvaluator(Engine):
    def __init__(self,
                 dataloader: torch.utils.data.DataLoader,
                 metric: metrics.MetricCompose,
                 eval_fn: Callable=None,
                 tag: str='Eval',
                 progress: bool=False ):
        super( BasicEvaluator, self ).__init__()
        self.dataloader = dataloader
        self.metric = metric
        self.progress = progress
        self.add_callback( DefaultEvents.AFTER_STEP, callbacks=self._update_pbar)
        self._model = None
        self._tag = tag
        if not isinstance(eval_fn, Callable):
            eval_fn = BasicEvaluator.default_eval_fn
        self.eval_fn = eval_fn

    def _update_pbar(self, engine):
        if self.progress:
            self._pbar.update(1)

    def eval(self, model, device=None):
        device = device if device is not None else \
            torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        self._model = weakref.ref(model) # use weakref here
        self.device = device
        self.metric.reset()
        model.to(device)
        if self.progress:
            self._pbar = tqdm(total=len(self.dataloader), desc=self._tag)
        with torch.no_grad(), set_mode(model, training=False):
            super(BasicEvaluator, self).run( self.step_fn, self.dataloader, max_iter=len(self.dataloader) )
        if self.progress:
            self._pbar.close()
        return self.metric.get_results()
    
    @property
    def model(self):
        if self._model is not None:
            return self._model()
        return None

    def step_fn(self, engine, batch):
        batch = move_to_device(batch, self.device)
        self.eval_fn( engine, batch )
        
    @staticmethod
    def default_eval_fn(evaluator, batch):
        model = evaluator.model
        inputs, targets = split_batch(batch)
        outputs = model( inputs )
        evaluator.metric.update( outputs, targets )
        



