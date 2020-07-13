from .base import Callback
from typing import Callable, Union, Sequence
import weakref
import random
from kamal.utils import move_to_device, set_mode, split_batch, colormap
from kamal.core.attach import AttachTo
import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import math
import numbers

class VisualizeOutputs(Callback):
    def __init__(self, 
                 model,
                 dataset: torch.utils.data.Dataset, 
                 idx_list_or_num_vis: Union[int, Sequence]=5, 
                 normalizer: Callable=None,
                 prepare_fn: Callable=None,
                 decode_fn: Callable=None, # decode targets and preds
                 tag: str='viz'):

        self._dataset = dataset
        self._model = weakref.ref(model)
        if isinstance(idx_list_or_num_vis, int):
            self.idx_list = self._get_vis_idx_list(self._dataset, idx_list_or_num_vis)
        elif isinstance(idx_list_or_num_vis, Sequence):
            self.idx_list = idx_list_or_num_vis
        self._normalizer = normalizer        
        self._decode_fn = decode_fn
        if prepare_fn is None:
            prepare_fn = VisualizeOutputs.get_prepare_fn()
        self._prepare_fn = prepare_fn
        self._tag = tag

    def _get_vis_idx_list(self, dataset, num_vis):
        return random.sample(list(range(len(dataset))), num_vis)
    
    @torch.no_grad()
    def __call__(self, trainer):
        if trainer.tb_writer is None:
            trainer.logger.warning("summary writer was not found in trainer")
            return
        device = trainer.device
        model = self._model()
        with torch.no_grad(), set_mode(model, training=False):
            for img_id, idx in enumerate(self.idx_list):
                batch = move_to_device(self._dataset[idx], device)
                batch = [ d.unsqueeze(0) for d in batch ]
                inputs, targets, preds = self._prepare_fn(model, batch)
                if self._normalizer is not None:
                    inputs = self._normalizer(inputs)
                inputs = inputs.detach().cpu().numpy()
                preds = preds.detach().cpu().numpy()
                targets = targets.detach().cpu().numpy()
                if self._decode_fn: # to RGB 0~1 NCHW
                    preds = self._decode_fn(preds)
                    targets = self._decode_fn(targets)
                inputs = inputs[0]
                preds = preds[0]
                targets = targets[0]
                trainer.tb_writer.add_images("%s-%d"%(self._tag, img_id), np.stack( [inputs, targets, preds], axis=0), global_step=trainer.state.iter)

    @staticmethod
    def get_prepare_fn(attach_to=None, pred_fn=lambda x: x):
        attach_to = AttachTo(attach_to)
        def wrapper(model, batch):
            inputs, targets = split_batch(batch)
            outputs = model(inputs)
            outputs, targets = attach_to(outputs, targets)
            return inputs, targets, pred_fn(outputs)
        return wrapper
    
    @staticmethod
    def get_seg_decode_fn(cmap=colormap(), index_transform=lambda x: x+1): # 255->0, 0->1,
        def wrapper(preds): 
            if len(preds.shape)>3:
                preds = preds.squeeze(1)
            out = cmap[ index_transform(preds.astype('uint8')) ]
            out = out.transpose(0, 3, 1, 2) / 255
            return out
        return wrapper

    @staticmethod
    def get_depth_decode_fn(max_depth, log_scale=True, cmap=plt.get_cmap('jet')):
        def wrapper(preds): 
            if log_scale:
                _max_depth = np.log( max_depth )
                preds = np.log( preds )
            else:
                _max_depth = max_depth
            if len(preds.shape)>3:
                preds = preds.squeeze(1)
            out = (cmap(preds.clip(0, _max_depth)/_max_depth)).transpose(0, 3, 1, 2)[:, :3]
            return out
        return wrapper

class VisualizeSegmentation(VisualizeOutputs):
    def __init__(
        self, model, dataset: torch.utils.data.Dataset, idx_list_or_num_vis: Union[int, Sequence]=5, 
        cmap = colormap(),
        attach_to: int=0,

        normalizer: Callable=None,
        prepare_fn: Callable=None,
        decode_fn: Callable=None,
        tag: str='seg'
    ):
        if prepare_fn is None:
            prepare_fn = VisualizeOutputs.get_prepare_fn(attach_to=attach_to, pred_fn=lambda x: x.max(1)[1])
        if decode_fn is None:
            decode_fn = VisualizeOutputs.get_seg_decode_fn(cmap=cmap, index_transform=lambda x: x+1)

        super(VisualizeSegmentation, self).__init__(
            model=model, dataset=dataset, idx_list_or_num_vis=idx_list_or_num_vis,
            normalizer=normalizer, prepare_fn=prepare_fn, decode_fn=decode_fn,
            tag=tag
        )

class VisualizeDepth(VisualizeOutputs):
    def __init__(
        self, model, dataset: torch.utils.data.Dataset, 
        idx_list_or_num_vis: Union[int, Sequence]=5, 
        max_depth = 10,
        log_scale = True,
        attach_to: int=0,

        normalizer: Callable=None,
        prepare_fn: Callable=None,
        decode_fn: Callable=None,
        tag: str='depth'
    ):
        if prepare_fn is None:
            prepare_fn = VisualizeOutputs.get_prepare_fn(attach_to=attach_to, pred_fn=lambda x: x)
        if decode_fn is None:
            decode_fn = VisualizeOutputs.get_depth_decode_fn(max_depth=max_depth, log_scale=log_scale)
        super(VisualizeDepth, self).__init__(
            model=model, dataset=dataset, idx_list_or_num_vis=idx_list_or_num_vis,
            normalizer=normalizer, prepare_fn=prepare_fn, decode_fn=decode_fn,
            tag=tag
        )