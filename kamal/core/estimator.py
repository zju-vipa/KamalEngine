import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys

from tqdm import tqdm
from ..utils import Visualizer
from ..metrics import MetrcisCompose
from ..losses import CriterionsCompose

# default keywords
_silent = False
_log_interval = 10
_total_epochs = 500
_ckpt_name = 'model'
_ckpt_dir = 'checkpoints'
_enable_vis = False
_port = '15556'
_env = 'main'
_val_loader = None
_metrics = None
_scheduler = None
_step_size = 100
_gamma = 0.1
_cuda = True
_gpu_id = None


def _default_prepare_inputs_and_targets(data, teachers):
    return data[0], data[1:]


class Estimator(object):
    def __init__(self, model, criterions, train_loader, teachers, restore_ckpt=None, **kargs):
        self.kargs = kargs
        self.silent = self.kargs.pop('silent', _silent)

        self.model = model
        self.teachers = teachers
        self.criterions = criterions

        if isinstance(self.criterions, (list, tuple)):
            self.criterions = MetrcisCompose(self.criterions)
        elif not isinstance(self.criterions, CriterionsCompose):
            self.criterions = CriterionsCompose([self.criterions])

        self.train_loader = train_loader

        self._setup_device()
        self._setup_training_state()
        self._setup_optimizer()
        self._setup_visdom()

        if restore_ckpt is not None:
            self.restore_from(restore_ckpt)

        self._setup_dataset()

        self.model = self.model.to(self.device)

    def restore_from(self, ckpt, ctx_ckpt=None):
        self.model.load_state_dict( torch.load(ckpt) )
        if ctx_ckpt is not None:
            ctx_ckpt = torch.load(ctx_ckpt)
            self.optimizer.load_state_dict(ctx_ckpt['optimizer_state'])
            if self.scheduler:
                self.scheduler.load_state_dict(ctx_ckpt['scheduler_state'])
            self.cur_epoch = ctx_ckpt['cur_epoch']
            self.cur_iter = ctx_ckpt['cur_iter']

    def save_model(self, ckpt_dir, ckpt_name, val_results=None):
        model_ckpt_path = os.path.join(ckpt_dir, '%s-%06d.pth' %
                                       (ckpt_name, self.cur_iter))
        torch.save(self.model.state_dict(), model_ckpt_path)
        ctx_ckpt_path = os.path.join(ckpt_dir, '%s-%06d-ctx.pth' %
                                     (ckpt_name, self.cur_iter))
        ctx_state = {
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": None if self.scheduler is None else self.scheduler.state_dict(),
            "cur_epoch": self.cur_epoch,
            "cur_iter": self.cur_iter,
            "val_results": val_results
        }
        torch.save(ctx_state, ctx_ckpt_path)
        self.log("Model saved as %s" % model_ckpt_path)

    def fit(self):
        """ Train model on given datasets
        """
        while self.cur_epoch < self.total_epochs:
            self.train_single_epoch()

            if self.do_validation and (self.cur_epoch+1) % self.val_interval == 0:
                results = self.validate()
            else:
                results = None

            if (self.cur_epoch+1) % self.ckpt_interval == 0:
                self.save_model(self.ckpt_dir, self.ckpt_name, results)

        return self.model

    @staticmethod
    def move_to(tensors, device):
        if isinstance(tensors, (list, tuple)):
            return [t.to(device) for t in tensors]
        elif isinstance(tensors, torch.Tensor):
            return tensors.device

    def log(self, content):
        if not self.silent:
            print(content)

    def validate(self):
        self.log("validation...")
        self.metrics.reset()
        self.model.eval()
        for batch_idx, data in enumerate(tqdm(self.val_loader)):
            data = self.move_to(data, self.device)

            outs = self.model(data[0])
            if not isinstance(outs, (list, tuple)):
                outs = [outs]
            self.metrics.update(outs, data[1:])
        results = self.metrics.get_results()

        for result in self.metrics.to_str(results):
            self.log(result)

        if self.enable_vis:
            for result in results:
                for k, v in result.items():
                    try:  # some results are not scalars
                        self.vis.vis_scalar(k, self.cur_epoch, v)
                    except:
                        pass
        return results

    def train_single_epoch(self):
        self.model.train()
        self.log("Epoch %d/%d" % (self.cur_epoch+1, self.total_epochs))
        if self.scheduler is not None:
            self.scheduler.step()

        for batch_idx, data in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            data = self.move_to(data, self.device)
            inputs, targets = self.prepare_inputs_and_targets(
                data, self.teachers)

            outs = self.model(inputs)
            loss_terms = self.criterions(outs, targets)
            loss = sum(loss_terms)

            loss.backward()
            self.optimizer.step()
            self.cur_iter += 1

            # Train log
            if not self.silent and (batch_idx+1) % self.log_interval == 0:
                log_str = 'Epoch %d/%d, Batch %d/%d, total_loss=%f' % (
                    self.cur_epoch+1, self.total_epochs, batch_idx+1, len(self.train_loader), loss)
                tags = self.criterions.tags
                if tags is not None:
                    for tag, loss_term in zip(tags, loss_terms):
                        log_str += ', %s=%f' % (tag, loss_term)
                        if self.enable_vis:
                            self.vis.vis_scalar(
                                tag, self.cur_iter, loss_term.detach().cpu().numpy())
                self.log(log_str)

        self.cur_epoch += 1

    def _setup_training_state(self):
        self.prepare_inputs_and_targets = self.kargs.pop(
            'prepare_inputs_and_targets', _default_prepare_inputs_and_targets)
        self.cur_epoch = 0
        self.cur_iter = 0
        self.log_interval = self.kargs.pop('log_interval', _log_interval)
        self.total_epochs = self.kargs.pop('total_epochs', _total_epochs)
        self.print_fn = self.kargs
        self.ckpt_name = self.kargs.pop('ckpt_name', _ckpt_name)
        self.ckpt_dir = self.kargs.pop('ckpt_dir', _ckpt_dir)
        self.ckpt_interval = self.kargs.pop('ckpt_interval', 10)
        self.val_interval = self.kargs.pop('val_interval', 1)
        if not os.path.exists(self.ckpt_dir):
            os.mkdir(self.ckpt_dir)

    def _setup_visdom(self):
        self.enable_vis = self.kargs.pop('enable_vis', _enable_vis)
        self.vis = None

        if self.enable_vis:
            self.port = self.kargs.pop('port', _port)
            self.env = self.kargs.pop('env', _env)

            self.vis = Visualizer(port=self.port, env=self.env)

    def _setup_dataset(self):
        assert isinstance(
            self.train_loader, torch.utils.data.DataLoader), "train_loader should be instance of torch.utils.data.DataLoader"
        self.val_loader = self.kargs.pop('val_loader', _val_loader)
        self.metrics = self.kargs.pop('metrics', _metrics)
        self.do_validation = (
            self.val_loader is not None and self.metrics is not None)

        if self.do_validation:
            if isinstance(self.metrics, (list, tuple)):
                self.metrics = MetrcisCompose(self.metrics)
            elif not isinstance(self.metrics, MetrcisCompose):
                self.metrics = MetrcisCompose([self.metrics])

    @staticmethod
    def get_parameters(model):
        for p in model.parameters():
            if p.requires_grad:
                yield p

    def _setup_optimizer(self):
        self.optimizer = self.kargs.pop('optimizer', None)

        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(params=self.get_parameters(
                self.model), lr=self.kargs.pop('lr', 1e-4))

        assert isinstance(self.optimizer, torch.optim.Optimizer)
        self.scheduler = self.kargs.pop('scheduler', _scheduler)
        if self.scheduler == None:
            step_size = self.kargs.pop('step_size', _step_size)
            gamma = self.kargs.pop('gamma', _gamma)
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size, gamma)

    def _setup_device(self):
        cuda = self.kargs.pop('cuda', _cuda)
        gpu_id = self.kargs.pop('gpu_id', _gpu_id)

        if gpu_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        self.device = torch.device('cuda' if (
            torch.cuda.is_available() and cuda == True) else 'cpu')
