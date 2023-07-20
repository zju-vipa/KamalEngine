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

from kamal.core.engine.engine import Engine
from kamal.core.engine.hooks import FeatureHook
from kamal.core import tasks
from kamal.utils import set_mode, move_to_device, split_batch
from kamal.core.engine.events import DefaultEvents
import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F

import typing, time
import numpy as np

def eval_task1_fn(evaluator, batch):
    model = evaluator.model
    inputs, targets = split_batch(batch)
    outputs = model.forward1( inputs )
    # ipdb.set_trace()
    if evaluator._tag in ('edge_occlusion', 'edge_texture', 'depth_euclidean'):
        outputs = (outputs + 1) / 2 * 65535
    else:
        outputs = (outputs + 1) /2 * 256
    evaluator.metric.update( outputs, targets )

def eval_task2_fn(evaluator, batch):
    model = evaluator.model
    inputs, targets = split_batch(batch)
    outputs = model.forward2( inputs )
    if evaluator._tag in ('edge_occlusion', 'edge_texture', 'depth_euclidean'):
        outputs = (outputs + 1) / 2 * 65535
    else:
        outputs = (outputs + 1) /2 * 256
    # ipdb.set_trace()
    evaluator.metric.update( outputs, targets )


class TaskonomyAmalgamation(Engine):

    def setup(
            self,
            student,
            dataloader:  torch.utils.data.DataLoader, 
            optimizer:   torch.optim.Optimizer, 
            alpha=1.5,
            device=None
        ):
        self._dataloader = dataloader
        if device is None:
            device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        self._device = device
        self._model = self._student = student.to(self._device)
        self._student.weights.requires_grad = True
        self._student.weight_fea.requires_grad = True
        self._student.weight_rep.requires_grad = True
        self._optimizer = optimizer
        self.alpha = alpha
        self.initial_task_loss = []

    @property
    def device(self):
        return self._device
    

    def run(self, max_iter, start_iter=0, epoch_length=None):
        self.state.iter = self._state.start_iter = start_iter
        self.state.max_iter = max_iter
        self.state.epoch_length = epoch_length if epoch_length else len(self._dataloader)
        self.state.dataloader = self._dataloader
        self.state.dataloader_iter = iter(self._dataloader)

        self.trigger_events(DefaultEvents.BEFORE_RUN)
        for self.state.iter in range( start_iter, max_iter ):
            if self.state.epoch_length!=None and \
                 self.state.iter%self.state.epoch_length==0: # Epoch Start
                    self.trigger_events(DefaultEvents.BEFORE_EPOCH)
        
            self.trigger_events(DefaultEvents.BEFORE_STEP)
            self.state.batch = self._get_batch()
            if self.state.iter < 3000 :
                step_output = self.init_step_fn(self, self.state.batch, self.state.iter)
                if isinstance(step_output, dict):
                    self.state.metrics.update(step_output)
            else:
                step_output = self.compe_step_fn(self, self.state.batch)
                if isinstance(step_output, dict):
                    self.state.metrics.update(step_output)
                step_output = self.col_step_fn(self, self.state.batch)
                if isinstance(step_output, dict):
                    self.state.metrics.update(step_output)
                step_output = self.ada_step_fn(self, self.state.batch)
                if isinstance(step_output, dict):
                    self.state.metrics.update(step_output)
          
            self.trigger_events(DefaultEvents.AFTER_STEP) 
              
            if self.state.epoch_length!=None and \
                 (self.state.iter+1)%self.state.epoch_length==0: # Epoch End
                    self.trigger_events(DefaultEvents.AFTER_EPOCH)
        self.trigger_events(DefaultEvents.AFTER_RUN)
        
    def compe_step_fn(self, engine, batch):
        start_time = time.perf_counter()
        data, t_data = batch
        data = move_to_device(data, self._device)
        t_data = move_to_device(t_data, self._device)
        t_out1, t_out2, t_enc_fea1, t_enc_fea2, t_enc_rep1, t_enc_rep2 = t_data
        s_out1, s_out2, s_enc_rep, s_enc_fea = self._student.forward_te( data )

        mse_loss = nn.MSELoss()
        loss_kd1 = mse_loss(s_out1, t_out1)
        loss_kd2 = mse_loss(s_out2, t_out2)
        loss_kd_sum = self._student.weights[0] * loss_kd1 + self._student.weights[1] * loss_kd2

        for param in self._student.encoder.parameters():
            param.requires_grad = False
        self._student.weights.requires_grad = False
        self._student.weight_fea.requires_grad = False
        self._student.weight_rep.requires_grad = False

        self._optimizer.zero_grad()
        loss_kd_sum.backward()
        self._optimizer.step()
        step_time = time.perf_counter() - start_time
        metrics = {'loss_kd1': loss_kd1, 'loss_kd2': loss_kd2}
        metrics.update({
            'total_loss': loss_kd_sum,
            'step_time': step_time,
            'lr': float( self._optimizer.param_groups[0]['lr'] )
        })
        return metrics
    
    def init_step_fn(self, engine, batch, iter_num):
        start_time = time.perf_counter()
        data, t_data = batch
        data = move_to_device(data, self._device)
        t_data = move_to_device(t_data, self._device)
        t_out1, t_out2, t_enc_fea1, t_enc_fea2, t_enc_rep1, t_enc_rep2 = t_data
        s_out1, s_out2, s_enc_rep, s_enc_fea = self._student.forward_te( data )

        mse_loss = nn.MSELoss()
        loss_kd1 = mse_loss(s_out1, t_out1)
        loss_kd2 = mse_loss(s_out2, t_out2)
        loss_kd_sum = self._student.weights[0] * loss_kd1 + self._student.weights[1] * loss_kd2

        if iter_num == 0:
            self.initial_task_loss.append(loss_kd1)
            self.initial_task_loss.append(loss_kd2)
            self.initial_task_loss = torch.stack(self.initial_task_loss)

        self._student.weights.requires_grad = False
        self._student.weight_fea.requires_grad = False
        self._student.weight_rep.requires_grad = False


        self._optimizer.zero_grad()
        loss_kd_sum.backward()
        self._optimizer.step()
        step_time = time.perf_counter() - start_time
        metrics = {'loss_kd1': loss_kd1.item(), 'loss_kd2': loss_kd2.item()}
        metrics.update({
            'total_loss_kd': loss_kd_sum.item(),
            'step_time': step_time,
            'lr': float( self._optimizer.param_groups[0]['lr'] )
        })
        return metrics
    

    
    def col_step_fn(self, engine, batch):
        start_time = time.perf_counter()
        data, t_data = batch
        data = move_to_device(data, self._device)
        t_data = move_to_device(t_data, self._device)
        t_out1, t_out2, t_enc_fea1, t_enc_fea2, t_enc_rep1, t_enc_rep2 = t_data
        s_out1, s_out2, s_enc_rep, s_enc_fea = self._student.forward_te( data )
        s_data = move_to_device((s_out1, s_out2, s_enc_fea, s_enc_rep), self._device)
        s_out1, s_out2, s_enc_fea, s_enc_rep = s_data

        mse_loss = nn.MSELoss()
        loss_kd1 = mse_loss(s_out1, t_out1)
        loss_kd2 = mse_loss(s_out2, t_out2)
        loss_kd_sum = self._student.weights[0] * loss_kd1 + self._student.weights[1] * loss_kd2
        # ipdb.set_trace()
        loss_la = mse_loss(s_enc_fea, t_enc_fea1 * self._student.weight_fea[0] + t_enc_fea2 * self._student.weight_fea[1]) 
        loss_la = loss_la + mse_loss(s_enc_rep, t_enc_rep1 * self._student.weight_rep[0] + t_enc_rep2 * self._student.weight_rep[1])

        loss_sum = 0.05 * loss_kd_sum + loss_la

        self._student.weights.requires_grad = False
        self._student.weight_fea.requires_grad = True
        self._student.weight_rep.requires_grad = True
        for param in self._student.decoder1.parameters():
            param.requires_grad = False
        for param in self._student.decoder2.parameters():
            param.requires_grad = False
        for param in self._student.encoder.parameters():
            param.requires_grad = True

        self._optimizer.zero_grad()
        loss_sum.backward()
        self._optimizer.step()
        step_time = time.perf_counter() - start_time
        metrics = {'loss_kd1': loss_kd1.item(), 'loss_kd2': loss_kd2.item(), 'loss_la': loss_la.item()}
        metrics.update({
            'total_loss_kd': loss_kd_sum.item(),
            'step_time': step_time,
            'lr': float( self._optimizer.param_groups[0]['lr'] )
        })
        return metrics
    
    def ada_step_fn(self, engine, batch):
        start_time = time.perf_counter()
        data, t_data = batch
        data = move_to_device(data, self._device)
        t_data = move_to_device(t_data, self._device)
        t_out1, t_out2, t_enc_fea1, t_enc_fea2, t_enc_rep1, t_enc_rep2 = t_data
        s_out1, s_out2, s_enc_rep, s_enc_fea = self._student.forward_te( data )

        mse_loss = nn.MSELoss()
        task_loss = []
        task_loss.append(mse_loss(s_out1, t_out1))
        task_loss.append(mse_loss(s_out2, t_out2))
        
        task_loss = torch.stack(task_loss)

        weighted_task_loss = torch.mul(self._student.weights, task_loss)

        # get the total loss
        loss = torch.sum(weighted_task_loss)

        
        # clear the gradients
        self._optimizer.zero_grad()
        self._student.weights.requires_grad = True
        self._student.weight_fea.requires_grad = True
        self._student.weight_rep.requires_grad = True
        for param in self._student.decoder1.parameters():
            param.requires_grad = True
        for param in self._student.decoder2.parameters():
            param.requires_grad = True
        for param in self._student.encoder.parameters():
            param.requires_grad = True

        # do the backward pass to compute the gradients for the whole set of weights
        # This is equivalent to compute each \nabla_W L_i(t)
        loss.backward(retain_graph=True)

        # initial_task_loss = self.initial_task_loss.cpu().detach().numpy()
        # set the gradients of w_i(t) to zero because these gradients have to be updated using the GradNorm loss
        #print('Before turning to 0: {}'.format(model.weights.grad))
        if self._student.weights.grad != None:
            self._student.weights.grad.data = self._student.weights.grad.data * 0.0
        #print('Turning to 0: {}'.format(model.weights.grad))
        W = self._student.get_last_shared_layer()
        
        norms = []
        for i in range(len(task_loss)):
        # get the gradient of this task loss with respect to the shared parameters
            gygw = torch.autograd.grad(task_loss[i], W.parameters(), retain_graph=True)
            # compute the norm
            norms.append(torch.norm(torch.mul(self._student.weights[i], gygw[0])))
        norms = torch.stack(norms)

        initial_task_loss = self.initial_task_loss.cpu().detach().numpy()

        if torch.cuda.is_available():
            loss_ratio = task_loss.data.cpu().numpy() / initial_task_loss
        else:
            loss_ratio = task_loss.data.numpy() / initial_task_loss
                # r_i(t)
        inverse_train_rate = loss_ratio / np.mean(loss_ratio)

        # compute the mean norm \tilde{G}_w(t) 
        if torch.cuda.is_available():
            mean_norm = np.mean(norms.data.cpu().numpy())
        else:
            mean_norm = np.mean(norms.data.numpy())
             

        # compute the GradNorm loss 
        # this term has to remain constant
        constant_term = torch.tensor(mean_norm * (inverse_train_rate ** self.alpha), requires_grad=False)
        if torch.cuda.is_available():
            constant_term = constant_term.cuda()
            #print('Constant term: {}'.format(constant_term))
            # this is the GradNorm loss itself
        grad_norm_loss = torch.sum(torch.abs(norms - constant_term))
        self._optimizer.zero_grad()
        #print('GradNorm loss {}'.format(grad_norm_loss))
        self._student.weights.grad = torch.autograd.grad(grad_norm_loss, self._student.weights, retain_graph=True)[0]

        self._optimizer.step()
        
        normalize_coeff = 1 / torch.sum(self._student.weights.data, dim=0)
        self._student.weights.data = self._student.weights.data * normalize_coeff
        # ipdb.set_trace()
        step_time = time.perf_counter() - start_time
        metrics = {'loss_kd1': task_loss[0].item(), 'loss_kd2': task_loss[1].item()}
        metrics.update({
            'total_loss_kd': loss.item(),
            'step_time': step_time,
            'lr': float( self._optimizer.param_groups[0]['lr'] )
        })
        return metrics






