from kamal.utils import set_mode, move_to_device
import weakref
import typing
from typing import Dict, Any, List,Callable
from kamal.core.engine.events import DefaultEvents, Event
import torch
import torch.nn as nn

import time
import numpy as np
import logging
import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn import Module, CrossEntropyLoss, ModuleDict
from torch.utils.tensorboard import SummaryWriter
from kamal.slim.distillation.kd import KDDistiller
from utils import validate, validate_LTB, adjust_learning_rate, adjust_learning_rate_stage2, AverageMeter,accuracy
from loss import get_loss_forward
class LTBDistiller(KDDistiller):
    def __init__( self, 
                  logger=None,
                  tb_writer=None):
        super(KDDistiller, self).__init__(logger=logger, tb_writer=tb_writer)

    def setup(self, cfg: Dict[str, Any], train_loader: DataLoader, val_loader: DataLoader, module_dict: ModuleDict,\
               criterion_dict: ModuleDict, optimizer: Optimizer,  device: torch.device,ckpt_dir,stage):
        self.cfg = cfg
        self.dataloader = train_loader
        self.val_loader = val_loader
        self.module_dict = module_dict
        self.model = self.student = self.module_dict["student"]
        self.teacher = self.module_dict["teacher"]
        self.criterion_dict = criterion_dict
        self.optimizer = optimizer
        if device is None:
            device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        self.device = device
        self.kddlogger = logging.getLogger("train")
        self.kddlogger.info("Start training...")
        self.best_acc = 0 
        self.student.to(self.device)
        self.teacher.to(self.device)
        self.ckpt_dir = ckpt_dir
        self.stage = stage
        if self.stage == 's1':
            self.stage_train = True
        elif self.stage == 's2':
            self.stage_train = False

    def run(self, max_iter, start_iter=0, epoch_length=None):
        self.state.iter = self._state.start_iter = start_iter
        self.state.max_iter = max_iter
        self.state.epoch_length = epoch_length if epoch_length else len(self.dataloader)
        self.state.dataloader = self.dataloader
        self.state.dataloader_iter = iter(self.state.dataloader)
        self.state.step_fn = self.step_fn

        # validate teacher accuracy
        teacher_acc, _, _ = validate(
            val_loader = self.val_loader,
            model = self.teacher,
            criterion = self.criterion_dict["cls"],
            device = self.device
        )
        self.kddlogger.info("Teacher accuracy: %.4f", teacher_acc)

        self.trigger_events(DefaultEvents.BEFORE_RUN)
        for self.state.iter in range( start_iter, max_iter ):
            if self.state.epoch_length!=None and \
                 self.state.iter%self.state.epoch_length==0: # Epoch Start
                    self.trigger_events(DefaultEvents.BEFORE_EPOCH)
                    self.epoch = (self.state.iter+1)//self.state.epoch_length + 1
                    self.before_epoch_define(self.epoch)
            self.trigger_events(DefaultEvents.BEFORE_STEP)
            self.state.batch = self._get_batch()
            step_output = self.step_fn(self.state.batch)
            if isinstance(step_output, dict):
                self.state.metrics.update(step_output)
            self.trigger_events(DefaultEvents.AFTER_STEP)        
            if self.state.epoch_length!=None and \
                 (self.state.iter+1)%self.state.epoch_length==0: # Epoch End
                    self.trigger_events(DefaultEvents.AFTER_EPOCH)
                    val_acc, val_acc_top5, val_loss = validate_LTB(
                                                cfg=self.cfg,
                                                val_loader=self.val_loader,
                                                model=self.module_dict["student"],
                                                criterion=self.criterion_dict["cls"],
                                                device=self.device,
                                                num_classes=self.cfg["dataset"]["num_classes"],
                                                t=self.cfg["training"]["t"],
                                                epoch=self.epoch,
                                                loss_method=self.cfg["model"]["loss_method"],
                                                stage=self.stage
                                            )
                    self.kddlogger.info(
                            "Epoch: %04d | %04d, acc: %.4f, loss: %.5f, val_acc: %.4f, val_acc_top5: %.4f, val_loss: %.5f",
                            self.epoch-1, self.cfg["training"]["epochs"],
                            self.top1.avg, self.losses.avg,
                            val_acc, val_acc_top5, val_loss
                        )
                     # save the best model
                    if val_acc > self.best_acc:
                        self.best_acc = val_acc
                        best_ep = self.epoch

                        save_file = os.path.join(self.ckpt_dir, "best.pth")
                        self.kddlogger.info("Saving the best model with acc: %.4f", self.best_acc)
                        torch.save(self.student.state_dict(), save_file)
                    self.kddlogger.info("Epoch: %04d | %04d, best acc: %.4f,", self.epoch-1, self.cfg["training"]["epochs"], self.best_acc)
        self.trigger_events(DefaultEvents.AFTER_RUN)

    def additional_kd_loss(self,batch):
        return batch[0].new_zeros(1)

    def step_fn(self, batch):

        student = self.module_dict["student"].train()
        teacher = self.module_dict["teacher"].eval()
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        start_time = time.perf_counter()
        
        preact = False
        if self.cfg["kd_loss"]["name"] == "ABLoss":
            preact = True

        logit_s = student(inputs, int(self.cfg["training"]["t"])/(self.epoch), self.stage_train)

        with torch.no_grad():
            logit_t = teacher(inputs, is_feat=False, preact=preact)
        
        feat_s = None
        feat_t = None 

        if self.cfg["model"]["task"] == 'mc':
            # print(logit.shape, target.shape)
            # return 0
            # cls + kl div
            loss_cls = self.criterion_cls(logit_s, targets.squeeze())
            loss_div = self.criterion_div(logit_s, logit_t)

            loss_kd = get_loss_forward(
                cfg=self.cfg,
                feat_s=feat_s,
                feat_t=feat_t,
                logit_s=logit_s,
                logit_t=logit_t,
                target=targets,
                criterion_kd=self.criterion_kd,
                module_dict=self.module_dict
            )

            loss = self.gamma * loss_cls + self.alpha * loss_div + self.beta * loss_kd
            acc1, acc5 = accuracy(logit_s, targets.squeeze(), topk=(1, 5))
            self.losses.update(loss.item(), inputs.shape[0])
            self.top1.update(acc1[0].item(), inputs.shape[0])
            self.top5.update(acc5[0].item(), inputs.shape[0])

            # loss = criterion(logit, target.squeeze())
            # acc1, acc5 = accuracy(logit, target.squeeze(), topk=(1, 5))
            # losses.update(loss.item(), x.shape[0])
            # top1.update(acc1[0], x.shape[0])
            # top5.update(acc5[0], x.shape[0])
            loss_avg = self.losses.avg
            top1_avg = self.top1.avg
            top5_avg = self.top5.avg
        # elif loss_method == 'nce':
        elif self.cfg["model"]["task"] == 'mt':

            loss_list = []
            acc1, acc5 = [], []
            print(len(logit_s), logit_s[0].shape, logit_s[1].shape)
            print(targets.shape)
            for j in range(len(logit_s)):
                print(logit_s[j].shape)
                print(logit_s)

                loss_list.append(self.criterion_dict["cls"](logit_s[j], targets[:, j]))
                acc1.append(accuracy(logit_s[j], targets[:, j], topk=(1, 1))[0])
                acc5.append(accuracy(logit_s[j], targets[:, j], topk=(1, 1))[1])

                self.losses[j].update(loss_list[j].item(), inputs.shape[0])
                self.top1[j].update(acc1[j], inputs.shape[0])
                self.top5[j].update(acc5[j], inputs.shape[0])

            self.losses_avg = [self.losses[k].avg for k in range(len(self.losses))]
            self.top1_avg = [self.top1[k].avg for k in range(len(self.top1))]
            self.top5_avg = [self.top5[k].avg for k in range(len(self.top5))]

            loss_avg = sum(self.losses_avg) / len(self.losses_avg)
            top1_avg = sum(self.top1_avg) / len(self.top1_avg)
            top5_avg = sum(self.top5_avg) / len(self.top5_avg)

            loss = sum(loss_list)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()    
        step_time = time.perf_counter() - start_time
        loss_dict = {"cls": loss_cls.item(),
                    "div": loss_div.item(),
                    "kd": loss_kd.item(),
                    }

        metrics = { loss_name: loss_value for (loss_name, loss_value) in loss_dict.items() }
        metrics.update({
            'total_loss': loss_avg,
            'top_1':top1_avg,
            'top_5':top5_avg,
            'step_time': step_time,})
        if self.state.iter % 10 == 0:
            self.kddlogger.info(
                    "Epoch: %3d|%3d, idx: %d, total iter: %d, loss: %.5f, acc@1: %.4f, acc@5: %.4f",
                    self.epoch-1, self.cfg["training"]["epochs"],
                    self.state.iter, self.state.max_iter,
                    loss_avg, top1_avg, top5_avg
                )
        return metrics
    
    def before_epoch_define(self,epoch):
        if self.stage == 's1':
            global_lr, branch_lr = adjust_learning_rate(
                        cfg_lr_global=self.cfg["training"]["lr_global"],
                        cfg_lr_branch=self.cfg["training"]["lr_branch"],
                        optimizer=self.optimizer,
                        epoch_current=epoch,
                        epoch_sum=self.cfg["training"]["epochs"]
                        )
            self.kddlogger.info("current global_lr: %.6f, current branch_lr: %.6f", global_lr, branch_lr)
        elif self.stage == 's2':
            global_lr = adjust_learning_rate_stage2(
                    optimizer=self.optimizer,
                    epoch_current=epoch
                    )
            self.kddlogger.info("current global_lr: %.6f", global_lr)
        self.gamma = self.cfg["kd"]["loss_weights"]["classify_weight"]
        self.alpha = self.cfg["kd"]["loss_weights"]["kd_weight"]
        self.beta = self.cfg["kd"]["loss_weights"]["other_loss_weight"]
        self.kddlogger.info(
            "Starting train one epoch with [gamma: %.5f, alpha: %.5f, beta: %.5f]...",
            self.gamma, self.alpha, self.beta
        )
        self.criterion_cls = self.criterion_dict["cls"]
        self.criterion_div = self.criterion_dict["div"]
        self.criterion_kd = self.criterion_dict["kd"]

        # if loss_method == 'nce':
        if self.cfg["model"]["task"] == 'mt':
            self.losses = [AverageMeter() for _ in range(self.cfg["dataset"]["num_classes"])]
            self.top1 = [AverageMeter() for _ in range(self.cfg["dataset"]["num_classes"])]
            self.top5 = [AverageMeter() for _ in range(self.cfg["dataset"]["num_classes"])]

        # elif loss_method =='ce':
        elif self.cfg["model"]["task"] == 'mc':
            self.losses = AverageMeter()
            self.top1 = AverageMeter()
            self.top5 = AverageMeter()
