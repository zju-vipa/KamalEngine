from .kd import KDDistiller
from kamal.core.loss import KDLoss

import torch.nn as nn
import torch._ops

import time

class HintDistiller(KDDistiller):
    def __init__(self, student, teacher, regressor, hint_layer=2, T=1.0, gamma=1.0, alpha=None, beta=None, logger=None, viz=None):
        super(HintDistiller, self).__init__(student, teacher, T=T, gamma=gamma, alpha=alpha, logger=logger, viz=viz)
        self.regressor = regressor

        self._hint_layer = hint_layer
        self._beta = beta

    def train( self, start_iter, max_iter, train_loader, optimizer, device=None ):
        if device is None:
            device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

        self.regressor.to(device)
        super(HintDistiller, self).train(start_iter, max_iter, train_loader, optimizer, device)

    def step(self):
        self.optimizer.zero_grad()
        start_time = time.perf_counter()

        self.student.train()
        self.teacher.eval()  
        self.regressor.train()
        try:
            data, targets = self._train_loader_iter.next()
        except StopIteration:
            self._train_loader_iter = iter(self.train_loader) # reset iterator
            data, targets = self._train_loader_iter.next()
        data, targets = data.to(self.device), targets.to(self.device)

        feat_s, s_out = self.student(data, is_feat=True)
        with torch.no_grad():
            feat_t, t_out = self.teacher(data, is_feat=True)
            feat_t = [f.detach() for f in feat_t]
        f_s = self.regressor(feat_s[self._hint_layer])
        f_t = feat_t[self._hint_layer]
        loss = self._gamma * nn.CrossEntropyLoss()(s_out, targets) + self._alpha * KDLoss(T=self._T, use_kldiv=True)(s_out, t_out) +self._beta * nn.MSELoss()(f_s, f_t)
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