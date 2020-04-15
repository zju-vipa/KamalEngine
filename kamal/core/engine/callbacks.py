import abc
import os
import torch
from tqdm import tqdm
from typing import Sequence, Iterable
import random

from .. import metrics
from .trainer import set_mode
import typing
import shutil, math

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

class ValidationCallback(CallbackBase):
    def __init__(self,
                 interval,
                 evaluator,
                 model_name     = 'model',
                 log_tag     = None,
                 save_model     = ('best', 'latest'),
                 ckpt_tag       = 'model',
                 ckpt_dir       = 'checkpoints',
                 weights_only   = True):
        
        self.interval = interval
        self.evaluator = evaluator
        self.save_model = save_model
        self.model_name = model_name
        self.log_tag = log_tag

        self._ckpt_dir = ckpt_dir
        self._ckpt_tag = ckpt_tag
        self._weights_only = weights_only

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
        model = getattr( trainer, self.model_name )
        results = self.evaluator.eval( model )  
        trainer.history.put_scalars( **results )
        trainer.logger.info( "[val %s] Iter %d/%d: %s"%(self.model_name if self.log_tag is None else self.log_tag, trainer.iter, trainer.max_iter, results) )
        
        # Visualization
        if trainer.viz is not None:
            for k, v in results.items():
                if isinstance(v, Iterable):  # skip non-scalar value
                    continue
                else:
                    if self.log_tag is not None:
                        log_tag = "%s:%s"%(self.log_tag, k)
                    else:
                        log_tag = "%s:%s"%(self.model_name, k)
                    trainer.viz.line([v, ], [trainer.iter, ], win=log_tag, update='append', opts={'title': log_tag})

        primary_metric = self.evaluator.metrics.PRIMARY_METRIC
        score = results[primary_metric]
        trainer.history.put_scalars( score=score )

        if self.save_model is not None:
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
            obj = model.state_dict() if self._weights_only else model
            for pth_path in pth_path_list:
                torch.save(obj, pth_path)
                trainer.logger.info("\t%s" % (pth_path))

    def final_save(self, ckpt_dir=None):
        trainer = self.trainer()
        model = getattr( trainer, self.model_name )
        if ckpt_dir is None:
            ckpt_dir = self._ckpt_dir
        if self.save_model is not None:
            if 'latest' in self.save_model:
                os.makedirs(ckpt_dir, exist_ok=True)
                shutil.copy2( self.latest_ckpt, os.path.join(ckpt_dir, "%s-latest.pth"% (self._ckpt_tag) )  )
            if 'best' in self.save_model:
                os.makedirs(ckpt_dir, exist_ok=True)
                shutil.copy2( self.best_ckpt, os.path.join(ckpt_dir, "%s-best.pth"% (self._ckpt_tag) )  )


class LoggingCallback(CallbackBase):
    def __init__(self, interval=10, names=('total_loss', 'lr' ), smooth_window_sizes=(10, None)):
        self.interval = interval
        self._names = names
        self._smooth_window_sizes = [ None for _ in names ] if smooth_window_sizes is None else smooth_window_sizes
        if isinstance( self._smooth_window_sizes, int):
            self._smooth_window_sizes = [ self._smooth_window_sizes for _ in names ]

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

            if trainer.viz:
                opts={'title': name, 
                      'showlegend': ( smooth is not None ) }
                trainer.viz.line([latest_value, ], [trainer.iter, ], win=name, name='latest', update='append', opts=opts )
                if smooth:
                    trainer.viz.line( [smoothed_value, ], [trainer.iter, ], win=name, name='smoothed', update='append', opts=opts )

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
        
        if isinstance( self.scheduler, typing.Sequence ):
            for sch in self.scheduler:
                sch.step()
        else:
            self.scheduler.step()


class VisualizeSegmentationCallBack(CallbackBase):
    def __init__(self, interval, dataset, idx_list_or_num_vis=5, model_name='model',
                        denormalizer=None, scale_to_255=True):
        self.interval = interval
        self.dataset = dataset
        self.model_name = model_name

        if isinstance( idx_list_or_num_vis, int ):
            self.idx_list = self._get_vis_idx_list( dataset, idx_list_or_num_vis )
        elif isinstance( idx_list_or_num_vis, Iterable ):
            self.idx_list = idx_list_or_num_vis
        
        self._denormalizer = denormalizer
        self._scale_to_255 = scale_to_255

    def _get_vis_idx_list( self, dataset, num_vis ):
        return random.sample( list( range( len( dataset ) ) ), num_vis )

    def after_step(self):
        trainer = self.trainer()  # get current chainer
        if trainer.iter == 0 or trainer.iter % self.interval != 0:
            return
        device = trainer.device
        model = getattr(trainer, self.model_name)
        with torch.no_grad(), set_mode(model, training=False):
            for img_id, idx in enumerate(self.idx_list):
                inputs, targets = self.dataset[ idx ]
                inputs, targets = inputs.unsqueeze(0).to(device), targets.unsqueeze(0).to(device)
                
                preds = model( inputs ).max(1)[1]

                if self._denormalizer is not None:
                    inputs = self._denormalizer(inputs)
                inputs = inputs.cpu().numpy()
                if self._scale_to_255:
                    inputs = (inputs*255)
                inputs = inputs.astype('uint8')

                preds = preds.detach().cpu().numpy().astype('uint8')
                targets = targets.detach().cpu().squeeze(1).numpy().astype('uint8')
                
                inputs = inputs[0]
                preds = self.dataset.decode_seg_to_color(preds).transpose(0, 3, 1, 2)[0]  # nhwc => nchw
                targets = self.dataset.decode_seg_to_color(targets).transpose(0, 3, 1, 2)[0]

                trainer.viz.images([inputs, preds, targets], nrow=3, win=("seg-%d" % img_id), opts={'title': str(img_id)})


class VisualizeHistoryImagesCallbacks(CallbackBase):
    def __init__(self, interval=1, names=None):
        self.interval = interval
        self._names = names

    def after_step(self):
        trainer = self.trainer()
        if trainer.iter == 0 or trainer.iter % self.interval != 0:
            return
        if trainer.viz is None:
            return

        if self._names is None:
            name_list = trainer.history.vis_data.keys()
        else:
            name_list = self._names
        
        for img_name in name_list:
            images = trainer.history.vis_data[img_name]
            N,C,H,W = images.shape
            nrow = N if N<=3 else int(math.ceil(math.sqrt(N)))
            trainer.viz.images(images, nrow=nrow, win=img_name, opts={'title': img_name})


class VisualizeGeneratorCallback(CallbackBase):
    def __init__(self, interval=1, model_name='generator', batch_size=16, denormalizer=None, win='generated'):
        self.interval = interval
        self.batch_size = batch_size
        self.denormalizer = denormalizer
        self._model_name=model_name
        self._win = win

    def after_step(self):
        trainer = self.trainer()
        if trainer.iter == 0 or trainer.iter % self.interval != 0:
            return
        if trainer.viz is None:
            return
        generator = getattr( trainer, self._model_name )
        z = torch.randn( self.batch_size, generator.nz ).to(device=trainer.device)
        with torch.no_grad():
            fake = generator(z) # n, c, h, w
        if self.denormalizer is not None:
            fake = self.denormalizer(fake)
        fake = fake.clamp(0,1).cpu().numpy()
        N,C,H,W = fake.shape
        nrow = int(math.ceil(math.sqrt(N)))
        trainer.viz.images(fake, nrow=nrow, win=self._win, opts={'title': self._win})