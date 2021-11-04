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

import torch
from torch.cuda import check_error
import torch.nn as nn
import torch.nn.functional as F

from kamal.core.engine.engine import Engine
from kamal.core import tasks
from kamal.utils import set_mode, move_to_device, split_batch

import time

class SubTaskAdvTrainer(Engine):
    def setup(
        self, 
        student,
        teacher,
        adv_head,
        dataloader:  torch.utils.data.DataLoader, 
        optimizer_sub:    torch.optim.Optimizer, 
        optimizer_adv:    torch.optim.Optimizer, 
        sub_list = [],
        adv_list = [],
        device = None,
    ):
        self._dataloader = dataloader
        self._sub_list = sub_list
        self._adv_list = adv_list
        
        if device is None:
            device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        self._device = device

        self._model = self._student = student.to(self._device)
        self._teacher = teacher.to(self._device)
        self._adv_head = adv_head.to(self._device) 
        self._optimizer_sub = optimizer_sub
        self._optimizer_adv = optimizer_adv

    @property
    def device(self):
        return self._device
    
    def run(self, max_iter, start_iter=0, epoch_length=None):

        with set_mode(self._student, training=True),\
             set_mode(self._adv_head, training=False),\
             set_mode(self._teacher, training=False):
            super( SubTaskAdvTrainer, self ).run(self.step_fn, self._dataloader, start_iter=start_iter, max_iter=max_iter, epoch_length=epoch_length)

    def step_fn(self, engine, batch):
        start_time = time.perf_counter()
        batch = move_to_device(batch, self._device)
        data = batch[0]
        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = input[0]
            return hook

        self._student.fc.register_forward_hook(get_activation('fc'))
        s_out = self._student( data )
        features = activation['fc']
        adv_out = self._adv_head(features)

        with torch.no_grad():
            t_out = self._teacher( data ).detach()
        t_out_sub = t_out[:,self._sub_list]
        t_out_adv = t_out[:,self._adv_list]

        loss_kd1 = tasks.loss.kldiv( s_out, t_out_sub)
        loss_kd2 = tasks.loss.kldiv( adv_out, t_out_adv)

        loss1 = loss_kd1 - 0.5 * loss_kd2
        self._optimizer_sub.zero_grad()
        loss1.backward()
        self._optimizer_sub.step()

        adv_out_ = self._adv_head(features.detach())
        loss_kd2 = tasks.loss.kldiv( adv_out_, t_out_adv)
        loss2 = 0.5 * loss_kd2
        self._optimizer_adv.zero_grad()
        loss2.backward()
        self._optimizer_adv.step()

        print("loss_sub: %.4f, loss_adv: %.4f" % (loss_kd1.item(), loss_kd2.item()))

        loss_dict = { 
                'loss_sub':     loss_kd1 - loss_kd2,
                'loss_adv':     loss_kd2
        }

        step_time = time.perf_counter() - start_time

        metrics = { loss_name: loss_value.item() for (loss_name, loss_value) in loss_dict.items() }
        metrics.update({
            'sub_loss': loss1.item(),
            'adv_loss': loss2.item(),
            'step_time': step_time,
            'lr': float( self._optimizer_sub.param_groups[0]['lr'] )
        })
        return metrics

class TransferHeadTrainer(Engine):
    def setup(
        self, 
        model,
        dataloader:  torch.utils.data.DataLoader, 
        optimizer:    torch.optim.Optimizer, 
        device = None,
        weights = [1., 0., 0.],
    ):
        self._dataloader = dataloader
        self._weights = weights
        
        if device is None:
            device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        self._device = device

        self._model = model.to(self._device)

        for name, param in self._model.named_parameters():
            if name.split(".")[0] not in ["fc"]:
                param.requires_grad = False

        self._optimizer = optimizer
        self.loss = nn.CrossEntropyLoss()

    @property
    def device(self):
        return self._device
    
    def run(self, max_iter, start_iter=0, epoch_length=None):

        with set_mode(self._model, training=True):
            super( TransferHeadTrainer, self ).run(self.step_fn, self._dataloader, start_iter=start_iter, max_iter=max_iter, epoch_length=epoch_length)

    def step_fn(self, engine, batch):
        start_time = time.perf_counter()

        batch = move_to_device(batch, self._device)
        inputs, targets = split_batch(batch)

        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                # 如果你想feature的梯度能反向传播，那么去掉 detach（）
                activation[name] = input[0]
            return hook

        self._model.fc.register_forward_hook(get_activation('fc'))
        outputs = self._model( inputs )
        features = activation['fc']
        features.detach()

        loss_entropy = self.loss( outputs, targets)
        loss_dict = { 
                'loss_entropy':     loss_entropy * self._weights[0],
        }
        loss = sum(loss_dict.values())
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        step_time = time.perf_counter() - start_time

        metrics = { loss_name: loss_value.item() for (loss_name, loss_value) in loss_dict.items() }
        metrics.update({
            'total_loss': loss.item(),
            'step_time': step_time,
            'lr': float( self._optimizer.param_groups[0]['lr'] )
        })
        return metrics