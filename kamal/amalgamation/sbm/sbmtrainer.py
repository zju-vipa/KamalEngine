from ...core import engine
from ...utils import set_mode

import torch
import torch.nn as nn

import typing
import time
import numpy as np 

class SbmTrainer(engine.trainer.MultitaskTrainer):
    def __init__(   self, 
                    task, 
                    model,
                    teachers,
                    split_size,
                    logger=None,
                    viz=None):
        super(SbmTrainer, self).__init__(task, model, teachers, split_size, logger=logger, viz=viz)
        
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
        data[0] = data[0].to(self.device) # move to device

        loss_dict = self.task.get_loss( self.model, self.teachers, data[0], self.split_size ) # get loss
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
