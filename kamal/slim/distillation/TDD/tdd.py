from kamal.core.tasks.loss import get_loss_forward
from kamal.utils import set_mode, move_to_device
import weakref
import typing
from typing import Dict, Any, List, Callable
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

from kamal.vision.datasets import get_dataset
from kamal.vision.models.classification import get_model

from kamal.optim import get_optimizer

# from kamal.utils._utils import str2bool, get_logger, preserve_memory
from kamal.utils._utils import make_deterministic
from kamal.utils._utils import AverageMeter, accuracy
from kamal.utils.validate import validate


class TDDistiller(KDDistiller):
    def __init__(self,
                 logger=None,
                 tb_writer=None):
        super(KDDistiller, self).__init__(logger=logger, tb_writer=tb_writer)

    def setup(self, cfg: Dict[str, Any], train_loader: DataLoader, val_loader: DataLoader, module_dict: ModuleDict, \
              criterion_dict: ModuleDict, optimizer: Optimizer, lr_scheduler: MultiStepLR, device: torch.device,
              ckpt_dir):
        self.cfg = cfg
        self.dataloader = train_loader
        self.val_loader = val_loader
        self.module_dict = module_dict
        self.model = self.student = self.module_dict["student"]
        self.teacher = self.module_dict["teacher"]
        self.criterion_dict = criterion_dict
        self.optimizer = optimizer
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.kddlogger = logging.getLogger("train")
        self.kddlogger.info("Start training...")
        self.best_acc = 0
        self.global_step = 0
        self.lr_scheduler = lr_scheduler
        self.student.to(self.device)
        self.teacher.to(self.device)
        self.ckpt_dir = ckpt_dir

    def run(self, max_iter, start_iter=0, epoch_length=None):
        self.state.iter = self._state.start_iter = start_iter
        self.state.max_iter = max_iter
        self.state.epoch_length = epoch_length if epoch_length else len(self.dataloader)
        self.state.dataloader = self.dataloader
        self.state.dataloader_iter = iter(self.state.dataloader)
        self.state.step_fn = self.step_fn

        # validate teacher accuracy
        teacher_acc, _, _ = validate(
            val_loader=self.val_loader,
            model=self.module_dict["teacher"],
            criterion=self.criterion_dict["cls"],
            device=self.device
        )
        self.kddlogger.info("Teacher accuracy: %.4f", teacher_acc)

        self.trigger_events(DefaultEvents.BEFORE_RUN)
        for self.state.iter in range(start_iter, max_iter):
            if self.state.epoch_length != None and \
                    self.state.iter % self.state.epoch_length == 0:  # Epoch Start
                self.trigger_events(DefaultEvents.BEFORE_EPOCH)
                self.epoch = (self.state.iter + 1) // self.state.epoch_length + 1
                self.before_epoch_define(self.epoch)
            self.trigger_events(DefaultEvents.BEFORE_STEP)
            self.state.batch = self._get_batch()
            step_output = self.step_fn(self.state.batch)

            if isinstance(step_output, dict):
                self.state.metrics.update(step_output)
            self.trigger_events(DefaultEvents.AFTER_STEP)
            if self.state.epoch_length != None and \
                    (self.state.iter + 1) % self.state.epoch_length == 0:  # Epoch End
                self.trigger_events(DefaultEvents.AFTER_EPOCH)

                self.tb_writer.add_scalar("epoch/train_acc", self.top1.avg, self.epoch)
                self.tb_writer.add_scalar("epoch/train_loss", self.losses.avg, self.epoch)

                val_acc, val_acc_top5, val_loss = validate(
                    val_loader=self.val_loader,
                    model=self.module_dict["student"],
                    criterion=self.criterion_dict["cls"],
                    device=self.device
                )

                self.tb_writer.add_scalar("epoch/val_acc", val_acc, self.epoch)
                self.tb_writer.add_scalar("epoch/val_loss", val_loss, self.epoch)
                self.tb_writer.add_scalar("epoch/val_acc_top5", val_acc_top5, self.epoch)

                self.logger.info(
                    "Epoch: %04d | %04d, acc: %.4f, loss: %.5f, val_acc: %.4f, val_acc_top5: %.4f, val_loss: %.5f",
                    self.epoch, self.cfg["training"]["epochs"],
                    self.top1.avg, self.losses.avg,
                    val_acc, val_acc_top5, val_loss
                )

                self.lr_scheduler.step()

                # regular saving
                if self.epoch % self.cfg["training"]["save_ep_freq"] == 0:
                    self.logger.info("Saving epoch %d checkpoint...", self.epoch)
                    state = {
                        "epoch": self.epoch,
                        "model": self.module_dict["student"].state_dict(),
                        "acc": val_acc,
                        "optimizer": self.optimizer.state_dict(),
                        "lr_scheduler": self.lr_scheduler.state_dict()
                    }
                    save_file = os.path.join(self.ckpt_dir, "epoch_{}.pth".format(self.epoch))
                    torch.save(state, save_file)

                # save the best model
                if val_acc > self.best_acc:
                    self.best_acc = val_acc
                    best_ep = self.epoch

                    save_file = os.path.join(self.ckpt_dir, "best.pth")
                    self.logger.info("Saving the best model with acc: %.4f", self.best_acc)
                    torch.save(state, save_file)
                self.logger.info("Epoch: %04d | %04d, best acc: %.4f,", self.epoch - 1,
                                 self.cfg["training"]["epochs"], self.best_acc)
        self.trigger_events(DefaultEvents.AFTER_RUN)

    def step_fn(self, batch):
        start_time = time.perf_counter()

        self.global_step += 1

        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # ===================forward=====================
        preact = False
        if self.cfg["kd_loss"]["name"] == "ABLoss":
            preact = True

        feat_s, logit_s = self.model_s(inputs, is_feat=True, preact=preact)

        with torch.no_grad():
            feat_t, logit_t = self.model_t(inputs, is_feat=True, preact=preact)
            # feat_t = [f.detach() for f in feat_t]

        # cls + kl div
        loss_cls = self.criterion_cls(logit_s, targets)
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

        acc1, acc5 = accuracy(logit_s, targets, topk=(1, 5))
        self.losses.update(loss.item(), inputs.shape[0])
        self.top1.update(acc1[0], inputs.shape[0])
        self.top5.update(acc5[0], inputs.shape[0])

        # ===================backward=====================
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # print info
        self.tb_writer.add_scalars(
            main_tag="train/acc",
            tag_scalar_dict={
                "@1": acc1,
                "@5": acc5,
            },
            global_step=self.global_step
        )
        self.tb_writer.add_scalars(
            main_tag="train/loss",
            tag_scalar_dict={
                "cls": loss_cls.item(),
                "div": loss_div.item(),
                "kd": loss_kd.item()
            },
            global_step=self.global_step
        )
        if (self.state.iter % self.state.epoch_length) % self.cfg["training"]["print_iter_freq"] == 0:
            self.logger.info(
                "Epoch: [%3d|%3d], idx: %d, total iter: %d, loss: %.5f, acc@1: %.4f, acc@5: %.4f",
                self.epoch, self.cfg["training"]["epochs"],
                (self.state.iter % self.state.epoch_length), self.global_step,
                self.losses.val, self.top1.val, self.top5.val
            )

        step_time = time.perf_counter() - start_time
        loss_dict = {"cls": loss_cls.item(),
                     "div": loss_div.item(),
                     "kd": loss_kd.item(),
                     }

        metrics = {loss_name: loss_value for (loss_name, loss_value) in loss_dict.items()}
        metrics.update({
            'total_loss': loss.item() * inputs.shape[0],
            'top_1': acc1,
            'top_5': acc5,
            'step_time': step_time, })
        return metrics

    def before_epoch_define(self, epoch):
        self.gamma = self.cfg["kd"]["loss_weights"]["classify_weight"]
        self.alpha = self.cfg["kd"]["loss_weights"]["kd_weight"]
        self.beta = self.cfg["kd"]["loss_weights"]["other_loss_weight"]

        self.logger.info(
            "Starting train one epoch with [gamma: %.5f, alpha: %.5f, beta: %.5f]...",
            self.gamma, self.alpha, self.beta
        )

        self.criterion_cls = self.criterion_dict["cls"]
        self.criterion_div = self.criterion_dict["div"]
        self.criterion_kd = self.criterion_dict["kd"]

        for name, module in self.module_dict.items():
            if name == "teacher":
                module.eval()
            else:
                module.train()

        self.model_s = self.module_dict["student"]
        self.model_t = self.module_dict["teacher"]

        self.losses = AverageMeter()
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()