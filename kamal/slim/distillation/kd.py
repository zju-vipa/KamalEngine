from kamal.core.engine.trainer import Engine
from kamal.core.tasks.loss import kldiv
import torch.nn.functional as F
from kamal.utils.logger import get_logger
from kamal.utils import set_mode, move_to_device
import weakref

import torch
import torch.nn as nn

import time
import numpy as np

class KDDistiller(Engine):
    def __init__( self, 
                  logger=None,
                  tb_writer=None):
        super(KDDistiller, self).__init__(logger=logger, tb_writer=tb_writer)

    def setup(self, student, teacher, dataloader, optimizer, T=1.0, alpha=1.0, beta=1.0, gamma=1.0, device=None):
        self.model = self.student = student
        self.teacher = teacher
        self.dataloader = dataloader
        self.optimizer = optimizer
        if device is None:
            device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        self.device = device

        self.T = T
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

        self.student.to(self.device)
        self.teacher.to(self.device)

    def run( self, max_iter, start_iter=0, epoch_length=None):
        with set_mode(self.student, training=True), \
             set_mode(self.teacher, training=False):
            super( KDDistiller, self ).run( self.step_fn, self.dataloader, start_iter=start_iter, max_iter=max_iter, epoch_length=epoch_length)

    def additional_kd_loss(self, engine, batch):
        return batch[0].new_zeros(1)

    def step_fn(self, engine, batch):
        student = self.student
        teacher = self.teacher
        start_time = time.perf_counter()
        batch = move_to_device(batch, self.device)
        inputs, targets = batch
        outputs = student(inputs)
        with torch.no_grad():
            soft_targets = teacher(inputs)
    
        loss_dict = { "loss_kld":        self.alpha * kldiv(outputs, soft_targets, T=self.T),
                      "loss_ce":         self.beta * F.cross_entropy( outputs, targets ),
                      "loss_additional": self.gamma * self.additional_kd_loss(engine, batch) }
        
        loss = sum( loss_dict.values() )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        step_time = time.perf_counter() - start_time
        metrics = { loss_name: loss_value.item() for (loss_name, loss_value) in loss_dict.items() }
        metrics.update({
            'total_loss': loss.item(),
            'step_time': step_time,
            'lr': float( self.optimizer.param_groups[0]['lr'] )
        })
        return metrics


    