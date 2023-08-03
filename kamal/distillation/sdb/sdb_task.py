import abc
import torch
import torch.nn as nn
import torch.nn.functional as F 
import sys
import typing
from typing import Callable, Dict, List, Any
from collections.abc import Mapping, Sequence
from kamal.core import metrics, exceptions
from kamal.core.attach import AttachTo

class Task(object):
    def __init__(self, name):
        self.name = name
    
    @abc.abstractmethod
    def get_loss( self, outputs, targets ) -> Dict:
        pass

    @abc.abstractmethod
    def predict(self, outputs) -> Any:
        pass

class SDBTask(Task):
    def __init__(self, 
                 name: str, 
                 loss_fn_ce: Callable, 
                 loss_fn_kd: Callable,
                 loss_fn_mse: Callable,
                 scaling: float=1.0, 
                 pred_fn: Callable=lambda x: x.max(1)[1],
                 attach_to=None):

        super(SDBTask, self).__init__(name)
        self._attach = AttachTo(attach_to)
        self.loss_fn_ce = loss_fn_ce
        self.loss_fn_kd = loss_fn_kd
        self.loss_fn_mse = loss_fn_mse
        self.pred_fn = pred_fn
        self.scaling = scaling

    def get_loss_tch_1(self, output_tch, output_tch_noisy, targets):
        output_tch, output_tch_noisy, targets = self._attach(output_tch, output_tch_noisy, targets)
        return { self.name + "loss_ntch": self.loss_fn_ce( output_tch, targets ) * self.scaling * 2 + self.loss_fn_ce( output_tch_noisy, targets ) * self.scaling}
    
    def get_adv_loss_1(self, output_stu, output_tch, output_tch_noisy, T):
        output_stu, output_tch, output_tch_noisy, T = self._attach( output_stu, output_tch, output_tch_noisy, T)
        return{self.name + "loss_adv_1": self.loss_fn_kd(F.log_softmax(output_stu/ T, dim=1), F.softmax(output_tch/ T, dim=1)) * (T * T) - self.loss_fn_kd(F.log_softmax(output_tch_noisy.detach(), dim=1),
                                      F.softmax(output_tch, dim=1))}
    
    def get_adv_loss_2(self, output_stu, output_tch_noisy, T):
        output_stu, output_tch_noisy, T = self._attach(output_stu, output_tch_noisy, T)
        return{self.name + "loss_adv_2": self.loss_fn_kd(F.softmax(output_stu/ T, dim=1),F.softmax(output_tch_noisy / T, dim=1))}
    
    def get_random_loss(self, output_stu, output_rad, output_tch_noisy, T):
        output_stu, output_rad, output_tch_noisy, T = self._attach(output_stu, output_rad, output_tch_noisy, T)
        return{self.name + "loss_random": self.loss_fn_mse(F.relu(output_stu), F.relu(output_tch_noisy)) * 0.01 - self.loss_fn_mse(F.relu(output_rad), F.relu(output_tch_noisy)) }

    def predict(self, outputs):
        outputs = self._attach(outputs)
        return self.pred_fn(outputs)
        
    def __repr__(self):
        rep = "Task: [%s loss_fn=%s scaling=%.4f attach=%s]"%(self.name, str(self.loss_fn_ce), self.scaling, self._attach)
        return rep

class KD_SDB_Task(Task):
    def __init__(self, 
                 name: str, 
                 loss_fn_kd: Callable, 
                 loss_fn_ce: Callable,
                 scaling:float=1.0, 
                 pred_fn: Callable=lambda x: x,
                 attach_to=None):
        super(KD_SDB_Task, self).__init__(name)
        self._attach = AttachTo(attach_to)
        self.loss_fn_kd = loss_fn_kd
        self.loss_fn_ce = loss_fn_ce
        self.pred_fn = pred_fn
        self.scaling = scaling

    def get_loss(self, output, output_teacher, noisy_output_teacher, T, params, target):
        output, output_teacher, noisy_output_teacher, T, params, target = self._attach(output, output_teacher, noisy_output_teacher, T, params, target)
        return { self.name: self.loss_fn_kd(F.log_softmax(output/T, dim=1),
                             F.softmax(noisy_output_teacher/T, dim=1)) * (params.alpha * T * T) * self.scaling + \
                             self.loss_fn_ce(output, target)* (1. - params.alpha) * self.scaling }

    def predict(self, outputs):
        outputs = self._attach(outputs)
        return self.pred_fn(outputs)
        
    def __repr__(self):
        rep = "Task: [%s loss_fn=%s scaling=%.4f attach=%s]"%(self.name, str(self.loss_fn), self.scaling, self._attach)
        return rep