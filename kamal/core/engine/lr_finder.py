import torch
import os
import tempfile, uuid
import contextlib
import numpy as np
from tqdm import tqdm
from kamal.core import callbacks
from kamal.core.engine import evaluator
from kamal.core.engine.events import DefaultEvents

class _ProgressCallback(callbacks.Callback):
    def __init__(self, max_iter, tag):
        self._tag = tag
        self._max_iter = max_iter
        self._pbar = tqdm(total=self._max_iter, desc=self._tag)

    def __call__(self, enigine):
        self._pbar.update(1)

    def reset(self):
        self._pbar = tqdm(total=self._max_iter, desc=self._tag)

class _LinearLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """ Linear Scheduler
    """
    def __init__(self,
                 optimizer,
                 lr_range,
                 max_iter,
                 last_epoch=-1):
        self.lr_range = lr_range
        self.max_iter = max_iter
        super(_LinearLRScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        r = self.last_epoch / self.max_iter
        return [self.lr_range[0] + (self.lr_range[1]-self.lr_range[0]) * r for base_lr in self.base_lrs]

class _ExponentialLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_range, max_iter, last_epoch=-1):
        self.lr_range = lr_range
        self.max_iter = max_iter
        super(_ExponentialLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        r = self.last_epoch / (self.max_iter - 1)
        return [self.lr_range[0] * (self.lr_range[1] / self.lr_range[0]) ** r for base_lr in self.base_lrs]

class _LRFinderCallback(object):
    def __init__(self,
                 model,
                 optimizer,
                 metric_name,
                 evaluator,
                 smooth_momentum):
        self._model = model
        self._optimizer = optimizer
        self._metric_name = metric_name
        self._evaluator = evaluator
        self._smooth_momentum = smooth_momentum
        self._records = []
    
    @property
    def records(self):
        return self._records
        
    def reset(self):
        self._records = []

    def __call__(self, trainer):      
        model = self._model
        optimizer = self._optimizer
        if self._evaluator is not None:
            results = self._evaluator.eval( model )
            score = float(results[ self._metric_name ])
        else:
            score = float(trainer.state.metrics[self._metric_name])
        
        if self._smooth_momentum>0 and len(self._records)>0:
            score = self._records[-1][1] * self._smooth_momentum + score * (1-self._smooth_momentum)
         
        lr = optimizer.param_groups[0]['lr']
        self._records.append( ( lr, score ) )


class LRFinder(object):

    def _reset(self):
        init_state = torch.load( self._temp_file )
        self.trainer.model.load_state_dict( init_state['model'] )
        self.trainer.optimizer.load_state_dict( init_state['optimizer'] )
        try: 
            self.trainer.reset()
        except: pass

    def adjust_learning_rate(self, optimizer, lr):
        for group in optimizer.param_groups:
            group['lr'] = lr

    def _get_default_lr_range(self, optimizer):
        if isinstance( optimizer, torch.optim.Adam ):
            return ( 1e-5, 1e-2 )
        elif isinstance( optimizer, torch.optim.SGD ):
            return ( 1e-3, 0.2 )
        else:
            return ( 1e-5, 0.5)
    
    def plot(self, polyfit: int=3, log_x=True):
        import matplotlib.pyplot as plt
        lrs = [ rec[0] for rec in self.records ]
        scores = [ rec[1] for rec in self.records ]
        
        if polyfit is not None:
            z = np.polyfit( lrs, scores, deg=polyfit )
            fitted_score = np.polyval( z, lrs )
    
        fig, ax = plt.subplots()
        ax.plot(lrs, scores, label='score')
        ax.plot(lrs, fitted_score, label='polyfit')

        if log_x:
            plt.xscale('log')

        ax.set_xlabel("Learning rate")
        ax.set_ylabel("Score")
        return fig

    def suggest( self, mode='min', skip_begin=10, skip_end=5, polyfit=None ):

        scores = np.array( [ self.records[i][1] for i in range( len(self.records) ) ] )
        lrs = np.array( [ self.records[i][0] for i in range( len(self.records) ) ] )
        if polyfit is not None:
            z = np.polyfit( lrs, scores, deg=polyfit )
            scores = np.polyval( z, lrs )

        grad = np.gradient( scores )[skip_begin:-skip_end]
        index = grad.argmin() if mode=='min' else grad.argmax() 
        index = skip_begin + index
        return index, self.records[index][0]

    def find(self, 
                optimizer,
                model,
                trainer, 
                metric_name='total_loss', 
                metric_mode='min', 
                evaluator=None,
                lr_range=[1e-4, 0.1], 
                max_iter=100, 
                num_eval=None, 
                smooth_momentum=0.9, 
                scheduler='exp', # exp
                polyfit=None, # None
                skip=[10, 5],
                progress=True):

        self.optimizer = optimizer
        self.model = model
        self.trainer = trainer
        # save init state
        _filename = str(uuid.uuid4())+'.pth'
        _tempdir = tempfile.gettempdir()
        self._temp_file = os.path.join(_tempdir, _filename)
        init_state = {
            'optimizer': optimizer.state_dict(),
            'model': model.state_dict()
        }
        torch.save(init_state, self._temp_file)

        if num_eval is None or num_eval > max_iter:
            num_eval = max_iter
        if lr_range is None:
            lr_range = self._get_default_lr_range(optimizer)
        
        interval = max_iter // num_eval
        if scheduler=='exp':
            lr_sched = _ExponentialLRScheduler(optimizer, lr_range, max_iter=max_iter)
        else:
            lr_sched = _LinearLRScheduler(optimizer, lr_range, max_iter=max_iter)

        self._lr_callback = callbacks.LRSchedulerCallback(schedulers=[lr_sched])
        self._finder_callback = _LRFinderCallback(model, optimizer, metric_name, evaluator, smooth_momentum)
        self.adjust_learning_rate( self.optimizer, lr_range[0] )
        with self.trainer.save_current_callbacks():
            trainer.add_callback(
                DefaultEvents.AFTER_STEP, callbacks=[ 
                    self._lr_callback,
                    _ProgressCallback(max_iter, '[LR Finder]') ])
            trainer.add_callback(
                DefaultEvents.AFTER_STEP(interval), callbacks=self._finder_callback)
            self.trainer.run( start_iter=0, max_iter=max_iter )

        self.records = self._finder_callback.records # get records
        index, best_lr = self.suggest(mode=metric_mode, skip_begin=skip[0], skip_end=skip[1], polyfit=polyfit)
        self._reset()
        del self.model, self.optimizer, self.trainer
        return best_lr