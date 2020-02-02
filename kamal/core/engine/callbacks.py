import abc
import os
import torch
from tqdm import tqdm
from typing import Sequence, Iterable

from .. import metrics
from ... import utils
from .ctx import eval_ctx, device_ctx


class CallbackBase(abc.ABC):
    def __init__(self):
        pass

    def before_train(self):
        pass

    def before_step(self):
        pass

    def after_step(self):
        pass

    def after_train(self):
        pass

    
class SimpleValidationCallback(CallbackBase):
    def __init__(self,
                 interval:     int,
                 val_loader:   torch.utils.data.DataLoader,
                 metrics:      metrics.StreamMetricsBase,
                 save_model:   Sequence = ('best', 'latest'),
                 ckpt_tag:     str = 'model',
                 ckpt_dir:     str = 'checkpoints',
                 weights_only: bool = True,
                 viz               = None):

        self.interval = interval
        self.metrics = metrics
        self.val_loader = val_loader
        self.save_model = save_model

        self._ckpt_dir = ckpt_dir
        self._ckpt_tag = ckpt_tag
        self._weights_only = weights_only
        self._viz = viz

        if self.save_model is not None:
            for save_type in self.save_model:
                assert save_type in ('best', 'latest', 'interval'), \
                    'save_model should be None or a subset of (\"best\", \"latest\", \"interval\")'
            os.makedirs(self._ckpt_dir, exist_ok=True)

        self._best_score = -99999
        self._best_ckpt = None
        self._latest_ckpt = None

    def after_step(self):
        trainer = self.trainer()
        current_iter = int(trainer.iter)
        if current_iter == 0 or current_iter % self.interval != 0:
            return
        task, model, device = trainer.task, trainer.model, trainer.device
        
        self.metrics.reset()
        with torch.no_grad(), eval_ctx(model):
            for i, (inputs, targets) in enumerate(tqdm(self.val_loader)):
                inputs, targets = inputs.to(device), targets.to(device)
                infer_dict = task.inference(model, inputs)
                self.metrics.update(preds=infer_dict['preds'], targets=targets)
        val_results = self.metrics.get_results()
        # history
        trainer.history.put_scalars( **val_results )

        if self._viz is not None:
            for k, v in val_results.items():
                if isinstance(v, Iterable):  # skip non-scalar value
                    continue
                else:
                    viz.line([v, ], [current_iter, ], win=k,
                                update='append', opts={'title': k})
        trainer.logger.info( "[Val] Iter %d: %s"%(current_iter, val_results) )

        if self.save_model is not None:
            primary_metric = self.metrics.PRIMARY_METRIC
            score = val_results[primary_metric]

            pth_path_list = []

            # interval model
            if 'interval' in self.save_model:
                pth_path = os.path.join(self._ckpt_dir, "%s-%08d-%s-%.3f.pth"
                                        % (self._ckpt_tag, current_iter, primary_metric, score))
                pth_path_list.append(pth_path)

            # the latest model
            if 'latest' in self.save_model:
                pth_path = os.path.join(self._ckpt_dir, "%s-latest-%08d-%s-%.3f.pth"
                                        % (self._ckpt_tag, current_iter, primary_metric, score))
                # remove existed weights
                if self._latest_ckpt is not None and os.path.exists(self._latest_ckpt):
                    os.remove(self._latest_ckpt)
                pth_path_list.append(pth_path)
                self._latest_ckpt = pth_path

            # the best model
            if 'best' in self.save_model and score > self._best_score:
                pth_path = os.path.join(self._ckpt_dir, "%s-best-%08d-%s-%.3f.pth" %
                                        (self._ckpt_tag, current_iter, primary_metric, score))
                # remove existed weights
                if self._best_ckpt is not None and os.path.exists(self._best_ckpt):
                    os.remove(self._best_ckpt)
                pth_path_list.append(pth_path)
                self._best_score = score
                self._best_ckpt = pth_path

            # save model
            trainer.logger.info("Model saved as:")
            obj = model.state_dict() if self._weights_only else model
            for pth_path in pth_path_list:
                torch.save(obj, pth_path)
                trainer.logger.info("\t%s" % (pth_path))


class LoggingCallback(CallbackBase):
    def __init__(self, interval=10, names=('total_loss', ), smooth_window_size=None, viz=None):
        self.interval = interval
        self._names = names
        self._smooth_window_size = smooth_window_size
        self._viz = viz

    def after_step(self):
        trainer = self.trainer()
        if trainer.iter == 0 or trainer.iter % self.interval != 0:
            return
        logger = trainer.logger
        history = trainer.history
        current_iter = trainer.iter

        num_batchs_per_epoch = len(trainer.train_loader)
        current_epoch = current_iter // num_batchs_per_epoch
        total_epoch = trainer.max_iter // num_batchs_per_epoch
        current_batch = current_iter % num_batchs_per_epoch

        # create log info
        info_str = "Iter %d/%d (Epoch %d/%d, Batch %d/%d)"

        for name in self._names:
            latest_value = history.get_scalar(name)
            smoothed_value = history.get_scalar(name, self._smooth_window_size) if self._smooth_window_size else latest_value
            info_str += " %s=%.4f" % ( name, smoothed_value )

            if self._viz:
                opts={'title': name, 'showlegend': ( self._smooth_window_size is not None ) }

                self._viz.line([latest_value, ], [current_iter, ], win=name, name='latest', 
                        update='append', opts=opts )
                if self._smooth_window_size:
                    self._viz.line( [smoothed_value, ], [current_iter, ], win=name, name='smoothed',
                        update='append', opts=opts )
                    
        logger.info(info_str % (
            trainer.iter,      trainer.max_iter,
            current_epoch,     total_epoch,
            current_batch,     num_batchs_per_epoch
        ))


class LRSchedulerCallback(CallbackBase):
    def __init__(self, scheduler, interval=1):
        self.scheduler = scheduler
        self.interval = interval

    def after_step(self):
        trainer = self.trainer()
        if trainer.iter == 0 or trainer.iter % self.interval != 0:
            return
        self.scheduler.step()


class SegVisualizationCallback(CallbackBase):
    def __init__(self, interval, data_loader, viz, norm_mean=None, norm_std=None, to_255=True):
        self.interval = interval

        self.data_loader = data_loader

        self._viz = viz
        self._norm_mean = norm_mean
        self._norm_std = norm_std
        self._to_255 = to_255

    def after_step(self):
        trainer = self.trainer()  # get current chainer
        task, model = trainer.task, trainer.model
        if trainer.iter == 0 or trainer.iter % self.interval != 0:
            return

        device = trainer.device
        img_id = 0

        with torch.no_grad(), eval_ctx(model):
            for i, (inputs, targets) in enumerate(self.data_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                infer_dict = task.inference(model, inputs)

                if self._norm_mean is not None and self._norm_std is not None:
                    inputs = utils.denormalize(
                        inputs.detach(), self._norm_mean, self._norm_std)
                if self._to_255:
                    inputs = inputs*255

                inputs = inputs.cpu().numpy().astype('uint8')
                preds = infer_dict['preds'].detach().cpu().numpy()
                targets = targets.detach().cpu().squeeze(1).numpy()

                dataset = self.data_loader.dataset
                if isinstance(dataset, torch.utils.data.Subset):
                    dataset = dataset.dataset

                if hasattr(dataset, 'decode_target'):
                    preds = dataset.decode_target(
                        preds).transpose(0, 3, 1, 2)  # N, C, H, W
                    targets = dataset.decode_target(
                        targets).transpose(0, 3, 1, 2)
                else:
                    preds = utils.DEFAULT_COLORMAP[preds].transpose(0, 3, 1, 2)
                    targets = utils.DEFAULT_COLORMAP[targets].transpose(0, 3, 1, 2)

                for input, pred, target in zip(inputs, preds, targets):
                    self._viz.images([input, pred, target],
                                     nrow=3,
                                     win=("segvis-%d" % img_id),
                                     opts={'title': str(img_id)})
                    img_id += 1
