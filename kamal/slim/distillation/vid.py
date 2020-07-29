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

import numpy as np
import time
import torch.nn as nn
import torch._ops
import torch.nn.functional as F
from .kd import KDDistiller
from kamal.utils import set_mode
from kamal.core.tasks.loss import KDLoss

class VIDDistiller(KDDistiller):
    def __init__(self, logger=None, tb_writer=None ):
        super(VIDDistiller, self).__init__( logger, tb_writer )

    def setup(self, student, teacher, dataloader, optimizer, regressor_l, T=1.0, alpha=1.0, beta=1.0, gamma=1.0, stu_hooks=[], tea_hooks=[], out_flags=[], device=None):
        super( VIDDistiller, self ).setup( 
            student, teacher, dataloader, optimizer, T=T, alpha=alpha, beta=beta, gamma=gamma, device=device )
        self.regressor_l = regressor_l
        self.stu_hooks = stu_hooks
        self.tea_hooks = tea_hooks
        self.out_flags = out_flags
        self.regressor_l = [regressor.to(self.device).train() for regressor in self.regressor_l]

    def additional_kd_loss(self, engine, batch):
        feat_s =  [f.feat_out if flag else f.feat_in for (f, flag) in zip(self.stu_hooks, self.out_flags)]
        feat_t = [f.feat_out.detach() if flag else f.feat_in for (f, flag) in zip(self.tea_hooks, self.out_flags)]
        g_s = feat_s[1:-1]
        g_t = feat_t[1:-1]
        return sum([c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, self.regressor_l)])

class VIDRegressor(nn.Module):
    def __init__(self,
                num_input_channels,
                num_mid_channel,
                num_target_channels,
                init_pred_var=5.0,
                eps=1e-5):
        super(VIDRegressor, self).__init__()

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(
                in_channels, out_channels,
                kernel_size=1, padding=0,
                bias=False, stride=stride)

        self.regressor = nn.Sequential(
            conv1x1(num_input_channels, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_target_channels),
        )
        self.log_scale = torch.nn.Parameter(
            np.log(np.exp(init_pred_var-eps)-1.0) * torch.ones(num_target_channels)
            )
        self.eps = eps

    def forward(self, input, target):
        # pool for dimentsion match
        s_H, t_H = input.shape[2], target.shape[2]
        if s_H > t_H:
            input = F.adaptive_avg_pool2d(input, (t_H, t_H))
        elif s_H < t_H:
            target = F.adaptive_avg_pool2d(target, (s_H, s_H))
        else:
            pass
        pred_mean = self.regressor(input)
        pred_var = torch.log(1.0+torch.exp(self.log_scale))+self.eps
        pred_var = pred_var.view(1, -1, 1, 1)
        neg_log_prob = 0.5*(
            (pred_mean-target)**2/pred_var+torch.log(pred_var)
            )
        loss = torch.mean(neg_log_prob)
        return loss
