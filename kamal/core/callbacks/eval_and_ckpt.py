from .base import Callback
import weakref
from kamal import utils
from typing import Sequence, Optional
import numbers
import os, shutil
import torch

class EvalAndCkpt(Callback):
    def __init__(self,
                 model,
                 evaluator,
                 metric_name:str,
                 metric_mode:str             ='max',
                 save_type:Optional[Sequence]=('best', 'latest'),
                 ckpt_dir:str                ='checkpoints',
                 ckpt_prefix:str             =None,
                 log_tag:str         ='model',
                 weights_only:bool   =True,
                 verbose:bool        =False,):

        self.metric_name = metric_name
        assert metric_mode in ('max', 'min'), "metric_mode should be 'max' or 'min'"
        self._metric_mode = metric_mode

        self._model = weakref.ref( model )
        self._evaluator = evaluator
        self._ckpt_dir = ckpt_dir
        self._ckpt_prefix = "" if ckpt_prefix is None else (ckpt_prefix+'_')
        if isinstance(save_type, str):
            save_type = ( save_type, )
        self._save_type = save_type
        if self._save_type is not None:
            for save_type in self._save_type:
                assert save_type in ('best', 'latest', 'all'), \
                    'save_type should be None or a subset of (\"best\", \"latest\", \"all\")'
        self._log_tag = log_tag
        self._weights_only = weights_only
        self._verbose = verbose

        self._best_score = -999999 if self._metric_mode=='max' else 99999.
        self._best_ckpt = None
        self._latest_ckpt = None

    @property
    def best_ckpt(self):
        return self._best_ckpt
    
    @property
    def latest_ckpt(self):
        return self._latest_ckpt

    @property
    def best_score(self):
        return self._best_score

    def __call__(self, trainer):
        model = self._model()
        results = self._evaluator.eval( model, device=trainer.device )  
        results = utils.flatten_dict(results)
        current_score = results[self.metric_name]

        scalar_results = { k: float(v) for (k, v) in results.items() if isinstance(v, numbers.Number) or len(v.shape)==0 }
        if trainer.logger is not None:
            trainer.logger.info( "[Eval %s] Iter %d/%d: %s"%(self._log_tag, trainer.state.iter, trainer.state.max_iter, scalar_results) )
        trainer.state.metrics.update( scalar_results )
        # Visualize results if trainer.tb_writer is not None

        if trainer.tb_writer is not None:
            for k, v in scalar_results.items():
                log_tag = "%s:%s"%(self._log_tag, k)
                trainer.tb_writer.add_scalar(log_tag, v, global_step=trainer.state.iter)

        if self._save_type is not None:
            pth_path_list = []
            # interval model
            if 'interval' in self._save_type:
                pth_path = os.path.join(self._ckpt_dir, "%s%08d_%s_%.3f.pth"
                                        % (self._ckpt_prefix, trainer.state.iter, self.metric_name, current_score))
                pth_path_list.append(pth_path)

            # the latest model
            if 'latest' in self._save_type:
                pth_path = os.path.join(self._ckpt_dir, "%slatest_%08d_%s_%.3f.pth"
                                        % (self._ckpt_prefix, trainer.state.iter, self.metric_name, current_score))
                # remove the old ckpt
                if self._latest_ckpt is not None and os.path.exists(self._latest_ckpt):
                    os.remove(self._latest_ckpt)
                pth_path_list.append(pth_path)
                self._latest_ckpt = pth_path

            # the best model
            if 'best' in self._save_type:
                if (current_score >= self._best_score and self._metric_mode=='max') or \
                    (current_score <= self._best_score and self._metric_mode=='min'):
                    pth_path = os.path.join(self._ckpt_dir, "%sbest_%08d_%s_%.4f.pth" %
                                            (self._ckpt_prefix, trainer.state.iter, self.metric_name, current_score))
                    # remove the old ckpt
                    if self._best_ckpt is not None and os.path.exists(self._best_ckpt):
                        os.remove(self._best_ckpt)
                    pth_path_list.append(pth_path)
                    self._best_score = current_score
                    self._best_ckpt = pth_path
            
            # save model
            if self._verbose and trainer.logger:
                trainer.logger.info("Model saved as:")
            obj = model.state_dict() if self._weights_only else model
            os.makedirs( self._ckpt_dir, exist_ok=True )
            for pth_path in pth_path_list:
                torch.save(obj, pth_path)
                if self._verbose and trainer.logger:
                    trainer.logger.info("\t%s" % (pth_path))
    
    def final_ckpt(self, ckpt_prefix=None, ckpt_dir=None):
        trainer = self.trainer()
        model = getattr(trainer, self.model_attr_name)

        if ckpt_dir is None:
            ckpt_dir = self._ckpt_dir
        if ckpt_prefix is None:
            ckpt_prefix = self._ckpt_prefix

        if self._save_type is not None:
            if 'latest' in self._save_type and self._latest_ckpt is not None:
                os.makedirs(ckpt_dir, exist_ok=True)
                shutil.copy2(self._latest_ckpt, os.path.join(ckpt_dir, "%slatest.pth"%ckpt_prefix))
            if 'best' in self._save_type and self._best_ckpt is not None:
                os.makedirs(ckpt_dir, exist_ok=True)
                shutil.copy2(self._best_ckpt, os.path.join(ckpt_dir, "%sbest.pth"%ckpt_prefix))