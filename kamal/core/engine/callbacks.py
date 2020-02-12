import abc
import os
import torch
from tqdm import tqdm
from typing import Sequence, Iterable
import random

from .. import metrics
from ...utils import denormalize
from .trainer import set_mode


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
                 interval,
                 evaluator,
                 save_model     = ('best', 'latest'),
                 ckpt_tag       = 'model',
                 ckpt_dir       = 'checkpoints',
                 weights_only   = True,
                 viz            = None):
        
        self.interval = interval
        self.evaluator = evaluator
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

        self.best_score = -99999
        self.best_ckpt = None
        self.latest_ckpt = None

    def after_step(self):
        trainer = self.trainer() # get the current trainer from weak reference
        if trainer.iter == 0 or trainer.iter % self.interval != 0:
            return   
        results = self.evaluator.eval( trainer.model )  
        trainer.history.put_scalars( **results )

        trainer.logger.info( "[Val] Iter %d/%d: %s"%(trainer.iter, trainer.max_iter, results) )

        # Visualization
        if self._viz is not None:
            for k, v in results.items():
                if isinstance(v, Iterable):  # skip non-scalar value
                    continue
                else:
                    self._viz.line([v, ], [trainer.iter, ], win=k,
                                update='append', opts={'title': k})

        if self.save_model is not None:
            primary_metric = self.evaluator.metrics.PRIMARY_METRIC
            score = results[primary_metric]
            pth_path_list = []

            # interval model
            if 'interval' in self.save_model:
                pth_path = os.path.join(self._ckpt_dir, "%s-%08d-%s-%.3f.pth"
                                        % (self._ckpt_tag, trainer.iter, primary_metric, score))
                pth_path_list.append(pth_path)

            # the latest model
            if 'latest' in self.save_model:
                pth_path = os.path.join(self._ckpt_dir, "%s-latest-%08d-%s-%.3f.pth"
                                        % (self._ckpt_tag, trainer.iter, primary_metric, score))
                # remove existed weights
                if self.latest_ckpt is not None and os.path.exists(self.latest_ckpt):
                    os.remove(self.latest_ckpt)
                pth_path_list.append(pth_path)
                self.latest_ckpt = pth_path

            # the best model
            if 'best' in self.save_model and score > self.best_score:
                pth_path = os.path.join(self._ckpt_dir, "%s-best-%08d-%s-%.3f.pth" %
                                        (self._ckpt_tag, trainer.iter, primary_metric, score))
                # remove existed weights
                if self.best_ckpt is not None and os.path.exists(self.best_ckpt):
                    os.remove(self.best_ckpt)
                pth_path_list.append(pth_path)
                self.best_score = score
                self.best_ckpt = pth_path

            # save model
            trainer.logger.info("Model saved as:")
            obj = trainer.model.state_dict() if self._weights_only else model
            for pth_path in pth_path_list:
                torch.save(obj, pth_path)
                trainer.logger.info("\t%s" % (pth_path))

class LoggingCallback(CallbackBase):
    def __init__(self, interval=10, names=('total_loss', 'lr' ), smooth_window_sizes=(10, None), viz=None):
        self.interval = interval
        self._names = names
        self._smooth_window_sizes = [ None for _ in names ] if smooth_window_sizes is None else smooth_window_sizes
        if isinstance( self._smooth_window_sizes, int):
            self._smooth_window_sizes = [ self._smooth_window_sizes for _ in names ]
        self._viz = viz

    def after_step(self):
        trainer = self.trainer()
        if trainer.iter == 0 or trainer.iter % self.interval != 0:
            return

        num_batchs_per_epoch = len(trainer.train_loader)
        total_epoch = trainer.max_iter // num_batchs_per_epoch
        current_epoch = trainer.iter // num_batchs_per_epoch
        current_batch = trainer.iter % num_batchs_per_epoch
        
        # create log info
        format_str = "Iter %d/%d (Epoch %d/%d, Batch %d/%d)"

        for name, smooth in zip(self._names, self._smooth_window_sizes):
            latest_value = trainer.history.get_scalar(name)
            smoothed_value = trainer.history.get_scalar(name, smooth) if smooth is not None else latest_value
            format_str += " %s=%.4f" % ( name, smoothed_value )

            if self._viz:
                opts={'title': name, 
                      'showlegend': ( smooth is not None ) }
                self._viz.line([latest_value, ], [trainer.iter, ], win=name, name='latest', update='append', opts=opts )
                if smooth:
                    self._viz.line( [smoothed_value, ], [trainer.iter, ], win=name, name='smoothed', update='append', opts=opts )

        trainer.logger.info(format_str % (
            trainer.iter,      trainer.max_iter,
            current_epoch,     total_epoch,
            current_batch,     num_batchs_per_epoch
        ))


class LRSchedulerCallback(CallbackBase):
    def __init__(self, interval=1, scheduler=None):
        self.scheduler = scheduler
        self.interval = interval

    def after_step(self):
        trainer = self.trainer()
        if self.scheduler is None or trainer.iter == 0 or trainer.iter % self.interval != 0:
            return
        self.scheduler.step()


class SegVisualizationCallback(CallbackBase):
    def __init__(self, interval, viz, dst, idx_list_or_num_vis=5, 
                        mean=None, std=None, scale_to_255=True):
        self.interval = interval
        self.dst = dst
        
        if isinstance( idx_list_or_num_vis, int ):
            self.idx_list = self._get_vis_idx_list( dst, idx_list_or_num_vis )
        elif isinstance( idx_list_or_num_vis, Iterable ):
            self.idx_list = idx_list_or_num_vis
        
        self._viz = viz
        self._mean = mean
        self._std = std
        self._scale_to_255 = scale_to_255

    def _get_vis_idx_list( self, dst, num_vis ):
        return random.sample( list( range( len( dst ) ) ), num_vis )

    def after_step(self):
        trainer = self.trainer()  # get current chainer
        if trainer.iter == 0 or trainer.iter % self.interval != 0:
            return
        device = trainer.device

        with torch.no_grad(), set_mode(trainer.model, training=False):
            for img_id, idx in enumerate(self.idx_list):
                inputs, targets = self.dst[ idx ]
                inputs, targets = inputs.unsqueeze(0).to(device), targets.unsqueeze(0).to(device)

                preds = trainer.model( inputs ).max(1)[1]

                if self._mean is not None and self._std is not None:
                    inputs = denormalize(inputs, self._mean, self._std)
                inputs = inputs.cpu().numpy()
                if self._scale_to_255:
                    inputs = (inputs*255)
                inputs = inputs.astype('uint8')

                preds = preds.detach().cpu().numpy().astype('uint8')
                targets = targets.detach().cpu().squeeze(1).numpy().astype('uint8')
                
                inputs = inputs[0]
                preds = self.dst.decode_seg_to_rgb(preds).transpose(0, 3, 1, 2)[0]  # nhwc => nchw
                targets = self.dst.decode_seg_to_rgb(targets).transpose(0, 3, 1, 2)[0]

                self._viz.images([inputs, preds, targets], nrow=3, win=("seg-%d" % img_id), opts={'title': str(img_id)})
