
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
from ..core import Estimator
from ..metrics import StreamClsMetrics, MetrcisCompose
from ..losses import SoftCELoss, CFLoss, CriterionsCompose


class CommonFeatureLearning(nn.Module):
    """Common Feature Learning Algorithm

    Learn common features from multiple pretrained networks.
    https://arxiv.org/abs/1906.10546

    **Parameters:**
        - **student** (nn.Module): student network
        - **teachers** (list or nn.ModuleList): a list of teacher networks
        - **cfl_blocks** (list or nn.ModuleList): common feature blocks like ``CFL_ConvBlock``
    """

    def __init__(self, student, teachers, cfl_blocks):
        super(CommonFeatureLearning, self).__init__()

        if not isinstance(teachers, nn.ModuleList):
            teachers = nn.ModuleList(teachers)
        self.student = student
        self.teachers = teachers
        self.teachers.eval()

        if not isinstance(cfl_blocks, nn.ModuleList):
            cfl_blocks = nn.ModuleList(cfl_blocks)

        self.cfl_blocks = cfl_blocks
        # fix params
        for p in self.teachers.parameters():
            p.requires_grad = False
    
    def train(self, mode=True):
        self.training = mode
        self.student.train(mode)
        self.cfl_blocks.train(mode)
        return self

    def fit(self, train_loader, T=1.0, sigmas=None, beta=10.0, **kargs):
        ce_loss = SoftCELoss(T=T)
        cf_loss = CFLoss(sigmas=[0.001, 0.01, 0.05, 0.1, 0.2, 1, 2], 
                         normalized=True)
        default_criterions = CriterionsCompose([ce_loss, cf_loss], weights=[1.0, beta], tags=['CE Loss', 'CF Loss'])

        criterions = kargs.pop("criterions", default_criterions)
        estimator = Estimator(model=self,
                              criterions=criterions,
                              train_loader=train_loader,
                              teachers=self.teachers,
                              prepare_inputs_and_targets=self.prepare_inputs_and_targets,
                              **kargs,
                              )
        estimator.fit()
        return self

    @staticmethod
    def prepare_inputs_and_targets(data, teachers):
        t_outs = []
        t_endpoints = []

        with torch.autograd.set_grad_enabled(False):
            for t in teachers:
                outs = t(data[0])
                endpoints = t.endpoints
                t_outs.append(outs)
                t_endpoints.append(endpoints)
        t_outs = torch.cat(t_outs, dim=1)
        return (data[0], t_endpoints), (t_outs, None)
    
    def s_forward(self, x):
        s_out = self.student(x)
        s_endpoints = self.student.endpoints
        return s_out, s_endpoints

    def cfl_forward(self, s_endpoints, t_endpoints, ):
        t_endpoints = list(zip(*t_endpoints))
        ka_tensor = []

        for i in range(len(self.cfl_blocks)):
            (hs, ht), (ft_, ft) = self.cfl_blocks[i](
                s_endpoints[i], t_endpoints[i])
            ka_tensor.append(((hs, ht), (ft_, ft)))
        return ka_tensor

    def forward(self, inputs):
        if self.training:
            x, t_endpoints = inputs
            s_outs, s_endpoints = self.s_forward(x)
            ka_tensor = self.cfl_forward(s_endpoints, t_endpoints)
            return s_outs, ka_tensor
        else:
            s_outs, s_endpoints = self.s_forward(inputs)
            return s_outs

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return {
                "student": self.student.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                "cfl_block": self.cfl_blocks.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars) 
        }

    def load_state_dict(self, state_dict, strict=True):
        self.student.load_state_dict(state_dict, strict)

    def parameters(self, recurse=True):
        return self.student.parameters(recurse)

    # def cfl_state_dict(self, destination=None, prefix='', keep_vars=False):
    #    return self.cfl_blocks.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    # def cfl_load_state_dict(self, state_dict, strict=True):
    #    self.cfl_blocks.load_state_dict(state_dict, strict)
#
    # def cflparameters(self, recurse=True):
    #    return self.cfl_blocks.parameters(recurse)
