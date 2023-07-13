import torch
import torch.nn as nn
from kamal.core.engine.engine import Engine, Event, DefaultEvents, State
from kamal.core import tasks,metrics, exceptions
from kamal.utils import set_mode, move_to_device, get_logger, split_batch
from typing import Callable, Mapping, Any, Sequence
import time
import weakref

class C2KDTrainer(Engine):
    def __init__( self, 
                  logger=None,
                  tb_writer=None):
        super(C2KDTrainer, self).__init__(logger=logger, tb_writer=tb_writer)

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
            super( C2KDTrainer, self ).run( self.step_fn, self.dataloader, start_iter=start_iter, max_iter=max_iter, epoch_length=epoch_length)

    def step_fn(self, engine, batch):
        model = self.model
        start_time = time.perf_counter()
        batch = move_to_device(batch, self.device)
        # target_lengths do not include the start_token and the end_token
        padded_input, padded_target, input_lengths, target_lengths, \
        enc_slf_attns_0, enc_slf_attns_3, dec_slf_attns_0, dec_slf_attns_3, dec_enc_attns, \
        tea_input_len, tea_target_len = split_batch(batch)

        outputs = self.model(padded_input=padded_input, input_lengths=input_lengths,
                                     padded_target=padded_target, target_lengths=target_lengths)
        preds, golds = outputs['pred'], outputs['gold']

        cer, wer = self.task.error_calculator(ys_hat=torch.argmax(preds, dim=2), ys_pad=golds)

        if self.task.ESA_criterion is not None:
            esa_loss_0 = self.task.ESA_criterion(stu_enc_slf=outputs['enc_slf_attn_0'], tea_enc_slf=enc_slf_attns_0,
                                                    stu_dec_enc=outputs['dec_enc_attn'], tea_dec_enc=dec_enc_attns,
                                                    cer=cer, wer=wer,
                                                    stu_input_len=input_lengths, tea_input_len=tea_input_len,
                                                    target_len=tea_target_len)

            esa_loss_3 = self.task.ESA_criterion(stu_enc_slf=outputs['enc_slf_attn_3'], tea_enc_slf=enc_slf_attns_3,
                                            stu_dec_enc=outputs['dec_enc_attn'], tea_dec_enc=dec_enc_attns,
                                            cer=cer, wer=wer,
                                            stu_input_len=input_lengths, tea_input_len=tea_input_len,
                                            target_len=tea_target_len)
            beta = self.task.ESA_criterion.beta
        else:
            esa_loss_0 = 0
            esa_loss_3 = 0
            beta = 0

        if self.task.DSA_criterion is not None:
            dsa_loss_0 = self.task.DSA_criterion(stu_dec_slf=outputs['dec_slf_attn_0'], tea_dec_slf=dec_slf_attns_0)
            dsa_loss_3 = self.task.DSA_criterion(stu_dec_slf=outputs['dec_slf_attn_3'], tea_dec_slf=dec_slf_attns_3)
            alpha = self.task.DSA_criterion.ratio
        else:
            alpha = 0
            dsa_loss_0 = 0
            dsa_loss_3 = 0

        ce_loss = self.task.ce_criterion(preds=preds, golds=golds, ratio=1 - alpha - beta)

        loss = ce_loss + dsa_loss_0 + dsa_loss_3 + esa_loss_0 + esa_loss_3
        loss_dict = {"C2KD":loss}

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