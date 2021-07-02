# Copyright 2020 Zhejiang Lab. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================

from .kd import KDDistiller
from kamal.core.tasks.loss import KDLoss

import torch.nn as nn
import torch._ops

import time


class HintDistiller(KDDistiller):
    def __init__(self, logger=None, tb_writer=None ):
        super(HintDistiller, self).__init__( logger, tb_writer )

    def setup(self, 
              student, teacher, regressor, dataloader, optimizer, 
              hint_layer=2, T=1.0, alpha=1.0, beta=1.0, gamma=1.0, 
              stu_hooks=[], tea_hooks=[], out_flags=[], device=None):
        super( HintDistiller, self ).setup( 
            student, teacher, dataloader, optimizer, T=T, alpha=alpha, beta=beta, gamma=gamma, device=device )
        self.regressor = regressor
        self._hint_layer = hint_layer
        self._beta = beta
        self.stu_hooks = stu_hooks
        self.tea_hooks = tea_hooks
        self.out_flags = out_flags
        self.regressor.to(device)
    
    def additional_kd_loss(self, engine, batch):
        feat_s = [f.feat_out if flag else f.feat_in for (f, flag) in zip(self.stu_hooks, self.out_flags)]
        feat_t = [f.feat_out.detach() if flag else f.feat_in for (f, flag) in zip(self.tea_hooks, self.out_flags)]
        f_s = self.regressor(feat_s[self._hint_layer])
        f_t = feat_t[self._hint_layer]
        return nn.functional.mse_loss(f_s, f_t)

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
