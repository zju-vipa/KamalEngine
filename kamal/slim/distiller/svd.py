from .kd import KDDistiller
from kamal.core.loss import SVDLoss
from kamal.core.loss import KDLoss

import torch
import torch.nn as nn

import time


class SVDDistiller(KDDistiller):
    def __init__(self, student, teacher, T=1.0, gamma=1.0, alpha=None, beta=None, stu_hooks=[], tea_hooks=[], out_flags=[], logger=None, viz=None):
        super(SVDDistiller, self).__init__(student, teacher,
                                                 T=T, gamma=gamma, alpha=alpha, logger=logger, viz=viz)
        self._beta = beta

        self.stu_hooks = stu_hooks
        self.tea_hooks = tea_hooks
        self.out_flags = out_flags

    def step(self):
        self.optimizer.zero_grad()
        start_time = time.perf_counter()

        self.student.train()
        self.teacher.eval()

        try:
            data, targets = self._train_loader_iter.next()
        except StopIteration:
            self._train_loader_iter = iter(self.train_loader)  # reset iterator
            data, targets = self._train_loader_iter.next()
        data, targets = data.to(self.device), targets.to(self.device)

        s_out = self.student(data)
        feat_s = [f.feat_out if flag else f.feat_in for (
            f, flag) in zip(self.stu_hooks, self.out_flags)]
        with torch.no_grad():
            t_out = self.teacher(data)
            feat_t = [f.feat_out.detach() if flag else f.feat_in for (
                f, flag) in zip(self.tea_hooks, self.out_flags)]
        g_s = feat_s[1:-1]
        g_t = feat_t[1:-1]
        loss = self._gamma * nn.CrossEntropyLoss()(s_out, targets) + self._alpha * \
            KDLoss(T=self._T, use_kldiv=True)(s_out, t_out) + \
            self._beta * SVDLoss()(g_s, g_t)
        loss.backward()

        # update weights
        self.optimizer.step()
        step_time = time.perf_counter() - start_time

        # record training info
        info = {'loss': loss}
        info['total_loss'] = float(loss.item())
        info['step_time'] = float(step_time)
        info['lr'] = float(self.optimizer.param_groups[0]['lr'])
        self._gather_training_info(info)
