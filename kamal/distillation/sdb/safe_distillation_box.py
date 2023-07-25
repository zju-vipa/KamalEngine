import torch
import torch.nn as nn
from kamal.core.engine.engine import Engine, Event, DefaultEvents, State
from kamal.core import tasks,metrics, exceptions
from kamal.utils import set_mode, move_to_device, get_logger, split_batch
from typing import Callable, Mapping, Any, Sequence
import time
import weakref

class AdversTeacher(Engine):
    def __init__( self, 
                  logger=None,
                  tb_writer=None):
        super(AdversTeacher, self).__init__(logger=logger, tb_writer=tb_writer)

    def setup(self, 
              model: torch.nn.Module, 
              advers_model: torch.nn.Module,
              random_model: torch.nn.Module,
              task: tasks.Task,
              dataloader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer, 
              params: None,
              noise: None,
              device: torch.device=None
              ):
        
        if device is None:
            device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        self.device = device
        if isinstance(task, Sequence):
            task = tasks.TaskCompose(task)
        self.task = task
        self.model = model
        self.advers_model = advers_model
        self.random_model = random_model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.params = params
        self.noise = noise
        return self

    def run( self, max_iter, start_iter=0, epoch_length=None):
        self.model.to(self.device)
        self.advers_model.to(self.device)
        self.random_model.to(self.device)
        with set_mode(self.model, training=True), \
        set_mode(self.advers_model, training=True), set_mode(self.random_model, training=True):
            super( AdversTeacher, self ).run( self.step_fn, self.dataloader, start_iter=start_iter, max_iter=max_iter, epoch_length=epoch_length)

    def step_fn(self, engine, batch):
        model = self.model
        advers_model = self.advers_model
        random_model = self.random_model
        params = self.params

        start_time = time.perf_counter()
        batch = move_to_device(batch, self.device)
        inputs, targets = split_batch(batch)
        noise = self.noise
        noisy_inputs = params.lamb * inputs + \
            (1-params.lamb) * noise.unsqueeze(0).repeat(inputs.size()[0],1,1,1)

        outputs = model(inputs)
        outputs_noisy = model(noisy_inputs)

        tch_loss_dict = self.task.get_loss_tch_1(outputs, outputs_noisy, targets)

        # computer pre-trained model output
        with torch.no_grad():
            output_stu = advers_model(inputs)  # logit without SoftMax
        output_stu = output_stu.detach()

        # knowledge disturbance loss
        T = params.temperature
        adv_loss1_dict = self.task.get_adv_loss_1(output_stu, outputs, outputs_noisy, T)
        adv_loss2_dict = self.task.get_adv_loss_2(output_stu, outputs_noisy, T)

        with torch.no_grad():
            output_rad = random_model(inputs)  # logit without SoftMax
        output_rad = output_rad.detach()
        random_loss_dict = self.task.get_random_loss(output_stu, output_rad, outputs_noisy, T)

        loss = sum( tch_loss_dict.values() ) - params.weight * sum( adv_loss1_dict.values() ) + sum( adv_loss2_dict.values() ) - params.eta * sum( random_loss_dict.values() )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_dict = {}
        loss_dict.update(tch_loss_dict)
        loss_dict.update(adv_loss1_dict)
        loss_dict.update(adv_loss2_dict)
        loss_dict.update(random_loss_dict)
        step_time = time.perf_counter() - start_time
        metrics = { loss_name: loss_value.item() for (loss_name, loss_value) in loss_dict.items() }
        metrics.update({
            'total_loss': loss.item(),
            'step_time': step_time,
            'lr': float( self.optimizer.param_groups[0]['lr'] )
        })
        return metrics

class KD_SDB_Stuednt(Engine):
    def setup(self, 
              student: torch.nn.Module, 
              teacher: torch.nn.Module, 
              task: tasks.Task,
              dataloader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer, 
              params: None,
              noise: None,
              device: torch.device=None):
        self.student = student
        self.teacher = teacher
        self.task = task
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.params = params
        self.noise = noise
        self.device = device
        return self

    def run( self, max_iter, start_iter=0, epoch_length=None):
        self.student.to(self.device)
        self.teacher.to(self.device)

        with set_mode(self.student, training=True), \
             set_mode(self.teacher, training=False):
            super(KD_SDB_Stuednt, self ).run(
                self.step_fn, self.dataloader, start_iter=start_iter, max_iter=max_iter, epoch_length=epoch_length)

    def step_fn(self, engine, batch):
        model = self.student
        teacher = self.teacher
        noise = self.noise
        params = self.params
        start_time = time.perf_counter()
        batch = move_to_device(batch, self.device)
        inputs, targets = split_batch(batch)
        noisy_inputs = params.lamb * inputs + \
                            (1-params.lamb) * noise.unsqueeze(0).repeat(inputs.size()[0],1,1,1)

        outputs = model(inputs)
        with torch.no_grad():
            output_teacher = teacher(inputs) 
            noisy_output_teacher = teacher(noisy_inputs) 
        T = params.temperature
        
        loss_dict = self.task.get_loss(outputs, output_teacher, noisy_output_teacher, T, params, targets)
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

class AdversTEvaluator(Engine):
    def __init__(self,
                 dataloader: torch.utils.data.DataLoader,
                 metric: metrics.MetricCompose,
                 params: None,
                 noise: None,
                 eval_fn: Callable=None,
                 tag: str='Eval',
                 progress: bool=False
                 ):
        super( AdversTEvaluator, self ).__init__()
        self.dataloader = dataloader
        self.metric = metric
        self.progress = progress
        if progress:
            self.porgress_callback = self.add_callback( 
                DefaultEvents.AFTER_STEP, callbacks=callbacks.ProgressCallback(max_iter=len(self.dataloader), tag=tag))
        self._model = None
        self._tag = tag
        if eval_fn is None:
            eval_fn = AdversTEvaluator.default_eval_fn
        self.eval_fn = eval_fn
        self.params = params
        self.noise = noise

    def eval(self, model, device=None):
        device = device if device is not None else \
            torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        self._model = weakref.ref(model) # use weakref here
        self.device = device
        self.metric.reset()
        model.to(device)
        if self.progress:
            self.porgress_callback.callback.reset()
        with torch.no_grad(), set_mode(model, training=False):
            super(AdversTEvaluator, self).run( self.step_fn, self.dataloader, max_iter=len(self.dataloader) )
        return self.metric.get_results()
    
    @property
    def model(self):
        if self._model is not None:
            return self._model()
        return None

    def step_fn(self, engine, batch):
        batch = move_to_device(batch, self.device)
        params = self.params
        noise = self.noise
        self.eval_fn( engine, batch, params, noise )
        
    def default_eval_fn(evaluator, batch, params, noise):
        model = evaluator.model
        inputs, targets = split_batch(batch)
        noisy_inputs = params.lamb * inputs + \
                                (1-params.lamb) * noise.unsqueeze(0).repeat(inputs.size()[0],1,1,1)
        outputs_noisy = model( noisy_inputs )
        evaluator.metric.update( outputs_noisy, targets )


