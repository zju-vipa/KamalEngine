import torch
import torch.nn as nn

import abc, math, weakref, typing, time
import numpy as np 

from . import hpo, history, task
from ...utils.logger import get_logger
from ...utils import set_mode

class TrainerBase(abc.ABC):
    def __init__(self, logger=None, viz=None):
        self._logger = logger if logger is not None else get_logger(name='kamal', color=True)
        self._viz = viz
        self.callbacks = []

    @property
    def logger(self):
        return self._logger

    @property
    def viz(self):
        return self._viz

    @property
    def max_iter(self):
        return self._max_iter
    
    @property
    def start_iter(self):
        return self._start_iter

    @property
    def iter(self):
        return self._iter
    
    def setup(self):
        return self
    
    def run(self, start_iter, max_iter):
        self._iter = start_iter
        self._start_iter, self._max_iter = start_iter, max_iter

        self.history = history.History(start_iter) # init history
        self.before_train()
        for self._iter in range( start_iter, max_iter ):
            self.before_step()
            self.step()
            self.after_step()
            self.history.step()
        self.after_train()

    def reset(self):
        self._iter = self.start_iter
        self.history = None
            
    def add_callbacks(self, callbacks: typing.Sequence):
        for callback in callbacks:
            callback.trainer = weakref.ref(self)
        self.callbacks.extend( callbacks )

    @abc.abstractmethod
    def step(self):
        pass

    def before_train(self):
        for callback in self.callbacks:
            callback.before_train()

    def after_train(self):
        for callback in self.callbacks:
            callback.after_train()

    def before_step(self):
        for callback in self.callbacks:
            callback.before_step()

    def after_step(self):
        for callback in self.callbacks:
            callback.after_step()
    

class BasicTrainer(TrainerBase):
    def __init__(   self, 
                    logger=None,
                    viz=None):
        super(BasicTrainer, self).__init__(logger, viz)

    def setup(self, 
              model:        nn.Module, 
              task:         task.TaskBase, 
              data_loader:  torch.utils.data.DataLoader, 
              optimizer:    torch.optim.Optimizer, 
              device        =None):
        """
        """
        self._data_loader = data_loader
        self._data_loader_iter = iter(data_loader)
        if device is None:
            device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        self._device = device

        self.model = model.to(self._device)
        self.optimizer = optimizer
        self.task = task
        return self

    def run( self, start_iter, max_iter ):
         with set_mode(self.model, training=True):
            super( BasicTrainer, self ).run( start_iter, max_iter)

    def reset(self):
        self._iter = self.start_iter
        self.history = None
        self._data_loader_iter = iter(data_loader)

    @property
    def data_loader(self):
        return self._data_loader

    @property
    def device(self):
        return self._device

    def _get_data(self):
        try:
            data = next( self._data_loader_iter )
        except StopIteration:
            self._data_loader_iter = iter(self._data_loader) # reset iterator
            data = next( self._data_loader_iter )
        if not isinstance( data, typing.Sequence ):
            data = [data, ]
        return data

    def step(self):
        self.optimizer.zero_grad()
        start_time = time.perf_counter()

        data = self._get_data()
        data = [ d.to(self._device) for d in data ] # move to device

        loss_list = self.task.get_loss( self.model, *data ) # get loss
        loss = sum( loss_list )
        loss.backward()
        self.optimizer.step()
        step_time = time.perf_counter() - start_time

        # record training info
        info = {
            'total_loss': loss.item(),
            'step_time': step_time,
            'lr': float( self.optimizer.param_groups[0]['lr'] )
        }
        self.history.put_scalars( **info )


class BasicKDTrainer(TrainerBase):
    def __init__(   self, 
                    logger=None,
                    viz=None, ):
        super( BasicKDTrainer, self ).__init__(logger, viz )

    def setup(self, student, teacher, task, data_loader, optimizer, device=None):
        self._data_loader = data_loader
        self._data_loader_iter = iter(data_loader)
        if device is None:
            device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        self._device = device

        self.model = self.student = student.to(self.device)
        self.teacher = teacher.to(self.device)
        self.optimizer = optimizer
        self.task = task
        return self

    def run( self, start_iter, max_iter ):
        with set_mode(self.student, training=True), \
             set_mode(self.teacher, training=False):
            super( BasicKDTrainer, self ).run( start_iter, max_iter)

    @property
    def data_loader(self):
        return self._data_loader

    @property
    def device(self):
        return self._device

    def _get_data(self):
        try:
            data = next( self._data_loader_iter )
        except StopIteration:
            self._data_loader_iter = iter(self._data_loader) # reset iterator
            data = next( self._data_loader_iter )
        if not isinstance( data, typing.Sequence ):
            data = [data, ]
        return data

    def step(self):
        self.optimizer.zero_grad()
        start_time = time.perf_counter()
        # prepare data
        data = self._get_data()
        data = [ d.to(self._device) for d in data ] # move to device

        with torch.no_grad():
            t_out = self.task.get_outputs( self.teacher, data[0] )
        loss_list = self.task.get_loss( self.student, data[0], [data[1], t_out] )
        loss = sum( loss_list )
        loss.backward()
        self.optimizer.step()
        step_time = time.perf_counter() - start_time

        # record training info
        info = {
            'total_loss': loss.item(),
            'step_time': step_time,
            'lr': float( self.optimizer.param_groups[0]['lr'] )
        }
        self.history.put_scalars( **info )


class MultitaskTrainer(BasicTrainer):
    def __init__(self, task, model, teachers, split_size, logger=None, viz=None):
        super(MultitaskTrainer, self).__init__(task, model, logger=logger, viz=viz)
        self.teachers = teachers
        self.split_size = split_size

    def train(self, start_iter, max_iter, train_loader, optimizer,  device=None):
        # init data_loader & optimizer
        if device is None:
            device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        self.device = device
        self.model.to(self.device)
        for i in range(len(self.teachers)):
            self.teachers[i].to(self.device)
        with set_mode(self.model, training=True):
            for i in range(len(self.teachers)):
                set_mode(self.teachers[i], training=False)
            super( MultitaskTrainer, self ).train(start_iter, max_iter, train_loader, optimizer, device=device)

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
            
        data[0] = data[0].to(self.device)# move to device
        data[1] = [d.to(self.device) for d in data[1]]
        loss_dict = self.task.get_loss( self.model, data[0], data[1], self.split_size ) # get loss
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
