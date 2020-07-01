from hyperopt import hp, fmin, tpe
import torch
from copy import deepcopy
from ruamel_yaml import YAML
import os
import tempfile
import uuid
import contextlib
import typing
import numpy as np
from tqdm import tqdm

from kamal.core.engine import evaluator, callbacks

@contextlib.contextmanager
def archive_callbasks(trainer):
    callbacks = trainer.callbacks
    trainer.callbacks = []
    yield
    trainer.callbacks = callbacks

def find_learning_rate(trainer, evaluator, lr_range=None, max_iter=400, num_eval=None, mode='min', progress=True):
    lr_finder = LRFinder(trainer)
    lr = lr_finder.find_lr(evaluator=evaluator, lr_range=lr_range, max_iter=max_iter, num_eval=num_eval, mode=mode, progress=progress )
    return lr

class _LRFinderScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self,
                 optimizer,
                 lr_range,
                 max_iter,
                 progress=False,
                 last_epoch=-1):
        self.lr_range = lr_range
        self.max_iter = max_iter
        self._pbar = tqdm(total=max_iter, desc='[LR Finder]') if progress else None

        super(_LRFinderScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        curr_iter = self.last_epoch + 1
        if self._pbar is not None:
            self._pbar.update(1)
        
        return [self.lr_range[0] + (self.lr_range[1]-self.lr_range[0]) * curr_iter / self.max_iter for group in self.optimizer.param_groups]

class _LRFinderCallback(callbacks.CallbackBase):
    def __init__(self,
                 interval,
                 evaluator,
                 model_name='model'):
        self._interval = interval
        self._evaluator = evaluator
        self._model_name = model_name
        self._records = []
        
    @property
    def records(self):
        return self._records

    def reset(self):
        self._records = []

    def after_step(self):
        trainer = self.trainer()  # get the current trainer from weak reference
        if trainer.iter == 0 or trainer.iter % self._interval != 0:
            return       
        model = getattr( trainer, self._model_name )
        optimizer = getattr( trainer, 'optimizer' )
        results = self._evaluator.eval( model )

        lr = optimizer.param_groups[0]['lr']
        score = results[ self._evaluator.PRIMARY_METRIC ]
        self._records.append( ( lr, score ) )

    def suggest( self, mode='min', skip_begin=2, skip_end=2 ):
        scores = np.array( [ self._records[i][1] for i in range( len(self.records) ) ] )[skip_begin:-skip_end]
        grad = np.gradient( scores )
        index = grad.argmin() if mode=='min' else grad.argmax() 
        index = skip_begin + index
        return index, self.records[index][0]

class LRFinder(object):
    def __init__(self, trainer, model_name='model'):
        self.trainer = trainer
        # save init state
        _filename = str(uuid.uuid4())+'.pth'
        _tempdir = tempfile.gettempdir()
        self._temp_file = os.path.join(_tempdir, _filename)
        init_state = {
            'optimizer': self.trainer.optimizer.state_dict(),
            'model': self.trainer.model.state_dict()
        }
        torch.save(init_state, self._temp_file)

    def _reset(self):
        init_state = torch.load( self._temp_file )
        self.trainer.model.load_state_dict( init_state['model'] )
        self.trainer.optimizer.load_state_dict( init_state['optimizer'] )
        try: 
            self.trainer.reset()
        except: 
            pass

    def _adjust_learning_rate(self, optimizer, lr):
        for group in optimizer.param_groups:
            group['lr'] = lr

    def _get_default_lr_range(self):
        if isinstance( self.trainer.optimizer, torch.optim.Adam ):
            return ( 1e-5, 1e-2 )
        elif isinstance( self.trainer.optimizer, torch.optim.SGD ):
            return ( 1e-3, 0.2 )
        else:
            return ( 1e-5, 0.5)
    
    def find_lr(self, evaluator, lr_range, max_iter, num_eval, mode='min', progress=True):
        if num_eval is None or num_eval > max_iter:
            num_eval = max_iter

        if lr_range is None:
            lr_range = self._get_default_lr_range()

        interval = max_iter // num_eval
        lr_sched = _LRFinderScheduler(self.trainer.optimizer, lr_range, max_iter=max_iter, progress=progress)
        self._finder_callback = _LRFinderCallback(interval=interval, evaluator=evaluator)
        self._lr_callback = callbacks.LRSchedulerCallback(interval=1, scheduler=[lr_sched])
        self._adjust_learning_rate( self.trainer.optimizer, lr_range[0] )

        with archive_callbasks(self.trainer):
            # use new callbacks
            self.trainer.add_callbacks([
                self._finder_callback,
                self._lr_callback,
            ])
            self.trainer.run( 0, max_iter )
        index, best_lr = self._finder_callback.suggest(mode=mode)
        self._reset()
        self._adjust_learning_rate( self.trainer.optimizer, lr=best_lr )
        return best_lr