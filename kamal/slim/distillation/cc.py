from .kd import KDDistiller
from kamal.core.criterions import KDLoss

import torch.nn as nn
import torch._ops

import time


class CCDistiller(KDDistiller):

    def __init__(self, logger=None, viz=None ):
        super(CCDistiller, self).__init__( logger, viz )

    def setup(self, student, teacher, data_loader, optimizer, embed_s, embed_t, T=1.0, gamma=1.0, alpha=None, beta=None, stu_hooks=[], tea_hooks=[], out_flags=[], device=None ):
        super(CCDistiller, self).setup(
            student, teacher, data_loader, optimizer, T=T, gamma=gamma, alpha=alpha, device=device)
        self.embed_s = embed_s.to(self.device).train()
        self.embed_t = embed_t.to(self.device).train()
        self._beta = beta
        self.stu_hooks = stu_hooks
        self.tea_hooks = tea_hooks
        self.out_flags = out_flags

    def step(self):
        self.optimizer.zero_grad()
        start_time = time.perf_counter()

        data, targets = self._get_data()
        data, targets = data.to(self.device), targets.to(self.device)

        s_out = self.student(data)
        feat_s = [f.feat_out if flag else f.feat_in for (
            f, flag) in zip(self.stu_hooks, self.out_flags)]
        with torch.no_grad():
            t_out = self.teacher(data)
            feat_t = [f.feat_out.detach() if flag else f.feat_in for (
                f, flag) in zip(self.tea_hooks, self.out_flags)]
        f_s = self.embed_s(feat_s[-1])
        f_t = self.embed_t(feat_t[-1])

        loss = self._gamma * nn.CrossEntropyLoss()(s_out, targets) + self._alpha * \
            KDLoss(T=self._T, use_kldiv=True)(s_out, t_out) + self._beta * \
            torch.mean((torch.abs(f_s-f_t)[:-1]
                        * torch.abs(f_s-f_t)[1:]).sum(1))
        loss.backward()

        # update weights
        self.optimizer.step()
        step_time = time.perf_counter() - start_time

        # record training info
        info = {'loss': loss}
        info['total_loss'] = float(loss.item())
        info['step_time'] = float(step_time)
        info['lr'] = float(self.optimizer.param_groups[0]['lr'])
        self.history.put_scalars(**info)


class LinearEmbed(nn.Module):
    """
        Linear Embedding
        @inproceedings{tian2019crd,
        title={Contrastive Representation Distillation},
        author={Yonglong Tian and Dilip Krishnan and Phillip Isola},
        booktitle={International Conference on Learning Representations},
        year={2020}
        }
    """

    def __init__(self, dim_in=1024, dim_out=128):
        super(LinearEmbed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x
