from .kd import KDDistiller
from kamal.core.tasks.loss import SVDLoss
from kamal.core.tasks.loss import KDLoss

import torch
import torch.nn as nn

import time


class SVDDistiller(KDDistiller):
    def __init__(self, logger=None, tb_writer=None ):
        super(SVDDistiller, self).__init__( logger, tb_writer )

    def setup(self, student, teacher, dataloader, optimizer, T=1.0, alpha=1.0, beta=1.0, gamma=1.0, stu_hooks=[], tea_hooks=[], out_flags=[], device=None):
        super( SVDDistiller, self ).setup( 
            student, teacher, dataloader, optimizer, T=T, alpha=alpha, beta=beta, gamma=gamma, device=device )
        self.stu_hooks = stu_hooks
        self.tea_hooks = tea_hooks
        self.out_flags = out_flags
        self._svd_loss = SVDLoss()

    def additional_kd_loss(self, engine, batch):
        feat_s = [f.feat_out if flag else f.feat_in for (f, flag) in zip(self.stu_hooks, self.out_flags)]
        feat_t = [f.feat_out.detach() if flag else f.feat_in for (f, flag) in zip(self.tea_hooks, self.out_flags)]
        g_s = feat_s[1:-1]
        g_t = feat_t[1:-1]
        return self._svd_loss( g_s, g_t )