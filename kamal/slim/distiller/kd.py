from kamal.core.engine.trainer import TrainerBase
from kamal.core.loss import KDLoss
from kamal.utils.logger import get_logger
from kamal.utils import comm
from kamal.core.engine.history import History

import torch
import torch.nn as nn

import time
import numpy as np

class KDDistiller(TrainerBase):
    def __init__(self, student, teacher, T=1.0, gamma =1.0 , alpha=None, logger=None, viz=None):
        super(KDDistiller, self).__init__( logger, viz )
        self.student = student
        self.teacher = teacher
        
        self._T = T
        self._gamma = gamma
        self._alpha = alpha

    def train( self, start_iter, max_iter, train_loader, optimizer, device=None ):
        self.train_loader = train_loader
        self.optimizer = optimizer
        self._train_loader_iter = iter(train_loader)
        if device is None:
            device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        self.device = device

        self.student.to(self.device)
        self.teacher.to(self.device)

        super( KDDistiller, self ).train( start_iter, max_iter)

    def step(self):
        self.optimizer.zero_grad()
        start_time = time.perf_counter()

        self.student.train()
        self.teacher.eval()   
        try:
            data, targets = self._train_loader_iter.next()
        except StopIteration:
            self._train_loader_iter = iter(self.train_loader) # reset iterator
            data, targets = self._train_loader_iter.next()
        data, targets = data.to(self.device), targets.to(self.device)
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
        self._gather_training_info( info )

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