
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
from ...core.loss import CFLLoss
from .blocks import CFL_ConvBlock
import kamal
from kamal import engine

class CommonFeatureLearning(nn.Module):
    """Common Feature Learning Algorithm

    Learn common features from multiple pretrained networks.
    https://arxiv.org/abs/1906.10546

    **Parameters:**
        - **layers** (list): student layer and teacher layers
        - **cfl_blocks** (list or nn.ModuleList): common feature blocks like ``CFL_ConvBlock``
    """
    def __init__(self, 
                 hooks, 
                 num_features,
                 cfl_block=None,
                 sigmas=[0.001, 0.01, 0.05, 0.1, 0.2, 1, 2],
                 beta=1.0):
        super(CommonFeatureLearning, self).__init__()
        self.hooks = hooks 
        self.cfl_block = CFL_ConvBlock(num_features[0], num_features[1:], 128) 
        self.cfl_criterion = CFLLoss( sigmas=sigmas, normalized=True, beta=beta)

    def forward(self, return_features=False):
        s_feature = self.hooks[0].feat_out
        t_features = [ h.feat_out.detach() for h in self.hooks[1:] ]
        (hs, hts), (fts_, fts) = self.cfl_block(s_feature, t_features)
        cfl_loss = self.cfl_criterion( hs, hts, fts_, fts ) 
        return cfl_loss

class CFLTrainer(engine.trainer.TrainerBase):
    def __init__(   self, 
                    student,
                    teachers,
                    cfl,
                    logger=None,
                    viz=None):
        self.student = self.model = student
        self.teachers = nn.ModuleList(teachers)
        self.cfl = cfl
        super(CFLTrainer, self).__init__(logger=logger, viz=viz)

    def train(self, start_iter, max_iter, train_loader, optim_s, optim_cfl, device=None):
        # init data_loader & optimizer
        self.train_loader = train_loader
        self.optim_s = optim_s
        self.optim_cfl = optim_cfl
        self._train_loader_iter = iter(train_loader)
        if device is None:
            device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        self.device = device

        self.student.to(self.device)
        self.teachers.to(self.device)

        with set_mode(self.student, training=True), \
             set_mode(self.teachers, training=False):
            super( CFLTrainer, self ).train( start_iter, max_iter )
    
    def step(self):
        self.optim_s.zero_grad()
        self.optim_cfl.zero_grad()

        start_time = time.perf_counter()

        try:
            data = next( self._train_loader_iter )
        except StopIteration:
            self._train_loader_iter = iter(self.train_loader) # reset iterator
            data = next( self._train_loader_iter )
        if isinstance( data, (list, tuple) ):
            data = data[0]
        data = data.to(self.device) # move to device

        s_out = self.student( data )
        t_outs = torch.cat( [teacher(data) for teacher in self.teachers], dim=1 )
        loss_kd = kamal.loss.kldiv( s_out, t_outs )
        loss_mmd = self.cfl()
        loss_dict = {'loss_kd': loss_kd, 'loss_mmd': loss_mmd}
        loss = sum( loss_dict.values() )
        loss.backward()

        # optimize
        self.optim_s.step()
        self.optim_cfl.step()
        step_time = time.perf_counter() - start_time

        # record training info
        info = loss_dict
        info['total_loss'] = loss
        info['step_time'] = step_time
        info['lr'] = float( self.optimizer.param_groups[0]['lr'] )
        self._gather_training_info( info )
