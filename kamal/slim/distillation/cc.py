from .kd import KDDistiller
from kamal.core.tasks.loss import KDLoss

import torch.nn as nn
import torch._ops

import time

class CCDistiller(KDDistiller):

    def __init__(self, logger=None, tb_writer=None ):
        super(CCDistiller, self).__init__( logger, tb_writer )

    def setup(self, student, teacher, dataloader, optimizer, embed_s, embed_t, T=1.0, alpha=1.0, beta=1.0, gamma=1.0, stu_hooks=[], tea_hooks=[], out_flags=[], device=None ):
        super(CCDistiller, self).setup(
            student, teacher, dataloader, optimizer, T=T, gamma=gamma, alpha=alpha, device=device)
        self.embed_s = embed_s.to(self.device).train()
        self.embed_t = embed_t.to(self.device).train()
        self.stu_hooks = stu_hooks
        self.tea_hooks = tea_hooks
        self.out_flags = out_flags

    def additional_kd_loss(self, engine, batch):
        feat_s = [f.feat_out if flag else f.feat_in for (f, flag) in zip(self.stu_hooks, self.out_flags)]
        feat_t = [f.feat_out.detach() if flag else f.feat_in for (f, flag) in zip(self.tea_hooks, self.out_flags)]
        f_s = self.embed_s(feat_s[-1])
        f_t = self.embed_t(feat_t[-1])
        return torch.mean((torch.abs(f_s-f_t)[:-1] * torch.abs(f_s-f_t)[1:]).sum(1))

class LinearEmbed(nn.Module):
    def __init__(self, dim_in=1024, dim_out=128):
        super(LinearEmbed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x
