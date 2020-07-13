from .kd import KDDistiller
from kamal.core.tasks.loss import RKDLoss
from kamal.core.tasks.loss import KDLoss

import torch
import torch.nn as nn

import time

class RKDDistiller(KDDistiller):
    def __init__(self, logger=None, tb_writer=None ):
        super(RKDDistiller, self).__init__( logger, tb_writer )

    def setup(self, student, teacher, dataloader, optimizer, T=1.0, alpha=1.0, beta=1.0, gamma=1.0, stu_hooks=[], tea_hooks=[], out_flags=[], device=None):
        super( RKDDistiller, self ).setup( 
            student, teacher, dataloader, optimizer, T=T, gamma=gamma, alpha=alpha, device=device )
        self.stu_hooks = stu_hooks
        self.tea_hooks = tea_hooks
        self.out_flags = out_flags
        self._rkd_loss = RKDLoss()
    
    def additional_kd_loss(self, engine, batch):
        feat_s = [f.feat_out if flag else f.feat_in for (f, flag) in zip(self.stu_hooks, self.out_flags)]
        feat_t = [f.feat_out.detach() if flag else f.feat_in for (f, flag) in zip(self.tea_hooks, self.out_flags)]
        f_s = feat_s[-1]
        f_t = feat_t[-1]
        return self._rkd_loss(f_s, f_t)
    