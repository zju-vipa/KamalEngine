import torch
import torch.nn as nn
from kamal.core.engine.engine import Engine, Event, DefaultEvents, State
from kamal.core import tasks
from kamal.utils import set_mode, move_to_device, get_logger, split_batch
from typing import Callable, Mapping, Any, Sequence
import time
import weakref

class BasicTrainer(Engine):
    def __init__( self, 
                  logger=None,
                  tb_writer=None):
        super(BasicTrainer, self).__init__(logger=logger, tb_writer=tb_writer)

    def setup(self, 
              model: torch.nn.Module, 
              task: tasks.Task,
              dataloader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer, 
              device: torch.device=None):
        
        if device is None:
            device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        self.device = device
        if isinstance(task, Sequence):
            task = tasks.TaskCompose(task)
        self.task = task
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        return self

    def run( self, max_iter, start_iter=0, epoch_length=None):
        self.model.to(self.device)
        with set_mode(self.model, training=True):
            super( BasicTrainer, self ).run( self.step_fn, self.dataloader, start_iter=start_iter, max_iter=max_iter, epoch_length=epoch_length)

    def step_fn(self, engine, batch):
        model = self.model
        start_time = time.perf_counter()
        batch = move_to_device(batch, self.device)
        inputs, targets = split_batch(batch)
        outputs = model(inputs)
        loss_dict = self.task.get_loss(outputs, targets) # get loss
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


class KDTrainer(BasicTrainer):

    def setup(self, 
              student: torch.nn.Module, 
              teacher: torch.nn.Module, 
              task: tasks.Task,
              dataloader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer, 
              device: torch.device=None):

        super(KDTrainer, self).setup(
            model=student, task=task, dataloader=dataloader, optimizer=optimizer, device=device)
        if isinstance(teacher, (list, tuple)):
            if len(teacher)==1:
                teacher=teacher[0]
            else:
                teacher = nn.ModuleList(teacher)
        self.student = self.model
        self.teacher = teacher
        return self

    def run( self, max_iter, start_iter=0, epoch_length=None):
        self.student.to(self.device)
        self.teacher.to(self.device)

        with set_mode(self.student, training=True), \
             set_mode(self.teacher, training=False):
            super( BasicTrainer, self ).run(
                self.step_fn, self.dataloader, start_iter=start_iter, max_iter=max_iter, epoch_length=epoch_length)

    def step_fn(self, engine, batch):
        model = self.model
        start_time = time.perf_counter()
        batch = move_to_device(batch, self.device)
        inputs, targets = split_batch(batch)
        outputs = model(inputs)
        if isinstance(self.teacher, nn.ModuleList):
            soft_targets = [ t(inputs) for t in self.teacher ]
        else:
            soft_targets = self.teacher(inputs)
        loss_dict = self.task.get_loss(outputs, soft_targets) # get loss
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
