import torch.nn as nn
import torch
import math
import abc
import weakref
import typing
import time
import numpy as np 

from ...utils.logger import get_logger
from ...utils import comm
from .history import History
from . import hpo

import contextlib

@contextlib.contextmanager
def set_mode(model, training=True):
    ori_mode = model.training
    model.train(training)
    yield
    model.train(ori_mode)

class TrainerBase(abc.ABC):
    def __init__(self, logger=None, viz=None):
        self._logger = logger if logger is not None else get_logger(name='kamal', color=True)
        self._viz = viz
        self._callbacks = []
        self._callbacks_enabled=True

    def set_callbacks(self, enable=True):
        self._callbacks_enabled = enable

    @property
    def callbacks(self):
        return self._callbacks

    @property
    def callbacks_enabled(self):
        return self._callbacks_enabled

    @property
    def logger(self):
        return self._logger

    @property
    def viz(self):
        return self._viz

    def reset(self):
        self.start_iter=self.start_iter
        self.history=None

    def train(self, start_iter, max_iter):
        self.iter = start_iter
        self.start_iter, self.max_iter = start_iter, max_iter

        self.history = History(start_iter) # init history
        self.before_train()
        for self.iter in range( start_iter, max_iter ):
            self.before_step()
            self.step()
            self.after_step()
            self.history.step()
        self.after_train()
            
    def add_callbacks(self, callbacks: typing.Sequence):
        for callback in callbacks:
            callback.trainer = weakref.ref(self)
        self._callbacks.extend( callbacks )

    @abc.abstractmethod
    def step(self):
        pass

    def before_train(self):
        if not self._callbacks_enabled: return
        for callback in self._callbacks:
            callback.before_train()

    def after_train(self):
        if not self._callbacks_enabled: return
        for callback in self._callbacks:
            callback.after_train()

    def before_step(self):
        if not self._callbacks_enabled: return
        for callback in self._callbacks:
            callback.before_step()

    def after_step(self):
        if not self._callbacks_enabled: return
        for callback in self._callbacks:
            callback.after_step()
    

class SimpleTrainer(TrainerBase):
    def __init__(   self, 
                    task, 
                    model,
                    logger=None,
                    viz=None):
        super(SimpleTrainer, self).__init__(logger, viz)
        self.task = task
        self.model = model
        
    def train(self, start_iter, max_iter, train_loader, optimizer,  device=None):
        # init data_loader & optimizer
        self.train_loader = train_loader
        self.optimizer = optimizer
        self._train_loader_iter = iter(train_loader)
        if device is None:
            device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        self.device = device
        self.model.to(self.device)

        with set_mode(self.model, training=True):
            super( SimpleTrainer, self ).train( start_iter, max_iter )
    
    def search_optimizer(self, evaluator, train_loader, hpo_space=None, mode='min', max_evals=20, max_iters=400):
        optimizer = hpo.search_optimizer(self, train_loader, evaluator=evaluator, hpo_space=hpo_space, mode=mode, max_evals=max_evals, max_iters=max_iters)
        return optimizer
    
    def step(self):
        self.optimizer.zero_grad()
        start_time = time.perf_counter()
        
        try:
            data = next( self._train_loader_iter )
        except StopIteration:
            self._train_loader_iter = iter(self.train_loader) # reset iterator
            data = next( self._train_loader_iter )
        if not isinstance( data, typing.Iterable ):
            data = [data, ]
        data = [ d.to(self.device) for d in data ] # move to device

        loss_dict = self.task.get_loss( self.model, *data ) # get loss
        loss = sum( loss_dict.values() )
        loss.backward()

        # optimize
        self.optimizer.step()
        step_time = time.perf_counter() - start_time

        # record training info
        info = loss_dict
        info['total_loss'] = loss
        info['step_time'] = step_time
        info['lr'] = float( self.optimizer.param_groups[0]['lr'] )
        self._gather_training_info( info )

    def reset(self):
        self.history = None
        self._train_loader_iter = iter(train_loader)
        self.iter = self.start_iter

    def _gather_training_info(self, info): 
        info = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in info.items()
        }
        current_lr = info.pop('lr')
        all_info = comm.gather(info)
        if comm.is_main_process():
            if "step_time" in all_info[0]:
                step_time = np.max([x.pop("step_time") for x in all_info])
                self.history.put_scalar("step_time", step_time)
                self.history.put_scalar("lr", current_lr)
            # average the rest training info
            info = {
                k: np.mean([x[k] for x in all_info]) for k in all_info[0].keys()
            }
            total_losses_reduced = sum(loss for loss in info.values())
            self.history.put_scalar("total_loss", total_losses_reduced)
            if len(info) > 1:
                self.history.put_scalars(**info)

class SimpleKDTrainer(SimpleTrainer):
    def __init__(   self, 
                    task, 
                    model,
                    teacher,
                    logger=None,
                    viz=None, ):

        super( SimpleKDTrainer, self ).__init__(task, model, logger, viz )
        self.teacher = teacher

    def train(self, start_iter, max_iter, train_loader, optimizer, device=None):
        if device is None:
            device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        self.device = device
        self.model.to(self.device)
        self.teacher.to(self.device)
        with set_mode(self.model, training=True), \
             set_mode(self.teacher, training=False):
            super( SimpleKDTrainer, self ).train( start_iter, max_iter, train_loader, optimizer, device=device)

    def step(self):
        self.optimizer.zero_grad()
        start_time = time.perf_counter()
        # prepare data
        try:
            data = next( self._train_loader_iter )
        except StopIteration:
            self._train_loader_iter = iter(self.train_loader)
            data = next( self._train_loader_iter )
        if not isinstance( data, typing.Sequence ):
            data = [data, ]
        data = [ d.to(self.device) for d in data ]

        # get loss
        loss_dict = self.task.get_loss( self.model, self.teacher, data[0] )
        loss = sum( loss_dict.values() )
        loss.backward()
        # update weights
        self.optimizer.step()
        step_time = time.perf_counter() - start_time

        # record training info
        info = loss_dict
        info['total_loss'] = float(loss.item())
        info['step_time'] = float(step_time)
        info['lr'] = float( self.optimizer.param_groups[0]['lr'] )
        self._gather_training_info( info )