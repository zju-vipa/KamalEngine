from .kd import KDDistiller
from kamal.core.tasks.loss import PKTLoss
from kamal.core.tasks.loss import KDLoss

import torch
import torch.nn as nn

import time

class PKTDistiller(KDDistiller):
    def __init__(self, logger=None, tb_writer=None ):
        super(PKTDistiller, self).__init__( logger, tb_writer )

    def setup(self, student, teacher, dataloader, optimizer, T=1.0, alpha=1.0, beta=1.0, gamma=1.0, stu_hooks=[], tea_hooks=[], out_flags=[], device=None):
        super( PKTDistiller, self ).setup( 
            student, teacher, dataloader, optimizer, T=T, alpha=alpha, beta=beta, gamma=gamma, device=device )
        self.stu_hooks = stu_hooks
        self.tea_hooks = tea_hooks
        self.out_flags = out_flags
        self._pkt_loss = PKTLoss()
    
    def additional_kd_loss(self, engine, batch):
        feat_s = [f.feat_out if flag else f.feat_in for (f, flag) in zip(self.stu_hooks, self.out_flags)]
        feat_t = [f.feat_out.detach() if flag else f.feat_in for (f, flag) in zip(self.tea_hooks, self.out_flags)]
        f_s = feat_s[-1]
        f_t = feat_t[-1]
        return self._pkt_loss(f_s, f_t)
