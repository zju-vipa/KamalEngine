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
from kamal.core.tasks.loss import AttentionLoss
from kamal.core.tasks.loss import KDLoss
from kamal.utils import set_mode, move_to_device

import torch
import torch.nn as nn

import time

class AttentionDistiller(KDDistiller):
    def __init__(self, logger=None, tb_writer=None ):
        super(AttentionDistiller, self).__init__( logger, tb_writer )
    
    def setup(self, 
              student, teacher, dataloader, optimizer, T=1.0, alpha=1.0, beta=1.0, gamma=1.0, 
              stu_hooks=[], tea_hooks=[], out_flags=[], device=None):
        super(AttentionDistiller, self).setup(
            student, teacher, dataloader, optimizer, T=T, alpha=alpha, beta=beta, gamma=gamma, device=device)
        self.stu_hooks = stu_hooks
        self.tea_hooks = tea_hooks
        self.out_flags = out_flags
        self._at_loss = AttentionLoss()
        
    def additional_kd_loss(self, engine, batch):
        feat_s = [f.feat_out if flag else f.feat_in for (f, flag) in zip(self.stu_hooks, self.out_flags)]
        feat_t = [f.feat_out.detach() if flag else f.feat_in for (f, flag) in zip(self.tea_hooks, self.out_flags)]
        g_s = feat_s[1:-1]
        g_t = feat_t[1:-1]
        return self._at_loss(g_s, g_t)
