from kamal.core.engine.trainer import TrainerBase
from kamal.core.criterions import KDLoss
from kamal.utils.logger import get_logger
from kamal.utils import set_mode
from kamal.core.engine.history import History

import torch
import torch.nn as nn

import time
import numpy as np

class KDDistiller(TrainerBase):
    def __init__(self, logger=None, viz=None ):
        super(KDDistiller, self).__init__( logger, viz )
    
    def setup(self, student, teacher, data_loader, optimizer, T=1.0, gamma=1.0, alpha=None, device=None):
        self.model = self.student = student
        self.teacher = teacher
        self._T = T
        self._gamma = gamma
        self._alpha = alpha

        self.data_loader = data_loader
        self.optimizer = optimizer
        self._data_loader_iter = iter(data_loader)
        if device is None:
            device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        self.device = device

        self.student.to(self.device)
        self.teacher.to(self.device)

    def _get_data(self):
        try:
            data = self._data_loader_iter.next()
        except StopIteration:
            self._data_loader_iter = iter(self.data_loader) # reset iterator
            data = self._data_loader_iter.next()
        return data

    def run( self, start_iter, max_iter):
        with set_mode(self.student, training=True), \
             set_mode(self.teacher, training=False):
            super( KDDistiller, self ).run( start_iter, max_iter)

    def step(self):
        self.optimizer.zero_grad()
        start_time = time.perf_counter()

        data, targets = self._get_data()
        data, targets = data.to(self.device), targets.to(self.device)
        with torch.no_grad():
            t_out = self.teacher( data )
        s_out = self.student( data )
        loss = self._gamma * nn.CrossEntropyLoss()(s_out, targets) + self._alpha * KDLoss(T=self._T, use_kldiv=True)(s_out, t_out)
        loss.backward()

        # update weights
        self.optimizer.step()
        step_time = time.perf_counter() - start_time
        
        # record training info
        info = {'loss': loss}
        info['total_loss'] = float(loss.item())
        info['step_time'] = float(step_time)
        info['lr'] = float( self.optimizer.param_groups[0]['lr'] )
        self.history.put_scalars( **info )

    