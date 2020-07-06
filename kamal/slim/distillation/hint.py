from .kd import KDDistiller
from kamal.core.criterions import KDLoss

import torch.nn as nn
import torch._ops

import time


class HintDistiller(KDDistiller):
    def __init__(self, logger=None, viz=None ):
        super(HintDistiller, self).__init__( logger, viz )

    def setup(self, 
              student, teacher, regressor, data_loader, optimizer, 
              hint_layer=2, T=1.0, gamma=1.0, 
              alpha=None, beta=None, 
              stu_hooks=[], tea_hooks=[], out_flags=[], device=None):
        super( HintDistiller, self ).setup( 
            student, teacher, data_loader, optimizer, T=T, gamma=gamma, alpha=alpha, device=device )
        self.regressor = regressor
        self._hint_layer = hint_layer
        self._beta = beta
        self.stu_hooks = stu_hooks
        self.tea_hooks = tea_hooks
        self.out_flags = out_flags
        self.regressor.to(device)
        
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
        f_s = self.regressor(feat_s[self._hint_layer])
        f_t = feat_t[self._hint_layer]
        loss = self._gamma * nn.CrossEntropyLoss()(s_out, targets) + self._alpha * \
            KDLoss(T=self._T, use_kldiv=True)(s_out, t_out) + \
            self._beta * nn.MSELoss()(f_s, f_t)
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


class Regressor(nn.Module):
    """
        Convolutional regression for FitNet
        @inproceedings{tian2019crd,
        title={Contrastive Representation Distillation},
        author={Yonglong Tian and Dilip Krishnan and Phillip Isola},
        booktitle={International Conference on Learning Representations},
        year={2020}
        }
    """

    def __init__(self, s_shape, t_shape, is_relu=True):
        super(Regressor, self).__init__()
        self.is_relu = is_relu
        _, s_C, s_H, s_W = s_shape
        _, t_C, t_H, t_W = t_shape
        if s_H == 2 * t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose2d(
                s_C, t_C, kernel_size=4, stride=2, padding=1)
        elif s_H >= t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=(1+s_H-t_H, 1+s_W-t_W))
        else:
            raise NotImplemented(
                'student size {}, teacher size {}'.format(s_H, t_H))
        self.bn = nn.BatchNorm2d(t_C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.is_relu:
            return self.relu(self.bn(x))
        else:
            return self.bn(x)
