import os
import abc
from typing import Any, Dict, List
import tqdm

import torch
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader
import torchvision.ops.boxes as box_ops

import cv_lib.distributed.utils as dist_utils
import cv_lib.metrics as metrics

from light_detr.utils import move_data_to_device
from light_detr.loss import SetCriterion
from light_detr.post_process import post_process
import light_detr.utils as detr_utils


class EvaluationBase(abc.ABC):
    """
    Distributed DETR evaluator
    """
    def __init__(
        self,
        criterion: SetCriterion,
        val_loader: DataLoader,
        loss_weights: Dict[str, float],
        device: torch.device
    ):
        self.main_process = dist_utils.is_main_process()
        self.criterion = criterion
        self.loss_weights = loss_weights
        self.val_loader = val_loader
        self.device = device
        self.metric: metrics.APMeter_Base = None

    def reset(self):
        self.metric.reset()

    def get_loss(self, output: Dict[str, Tensor], targets: List[Dict[str, Any]]):
        loss_dict: Dict[str, torch.Tensor] = self.criterion(output, targets)
        weighted_losses: Dict[str, torch.Tensor] = dict()
        for k, loss in loss_dict.items():
            k_prefix = k.split(".")[0]
            if k_prefix in self.loss_weights:
                weighted_losses[k] = loss * self.loss_weights[k_prefix]
        loss = sum(weighted_losses.values())
        loss = loss.detach()
        return loss, loss_dict

    def get_pred_bboxes(self, output: Dict[str, Tensor], targets: List[Dict[str, Any]]):
        target_sizes = torch.tensor([t["size"] for t in targets], device=self.device)
        pred_bboxes = post_process(output, target_sizes)
        return pred_bboxes

    @abc.abstractmethod
    def update_prediction(
        self,
        outputs: List[Dict[str, Tensor]],
        targets: List[Dict[str, Any]]
    ):
        pass

    def get_performance(self) -> Dict[str, float]:
        self.metric.accumulate()
        self.metric.sync()
        return self.metric.value()

    @staticmethod
    def decode_gt_box(targets: List[Dict[str, Any]]) -> List[Tensor]:
        """
        Convert gt_bboxes cxcywh -> xyxy w.r.t. image size after augmentation (same as prediction)
        """
        # (h, w)
        target_sizes = list(t["size"] for t in targets)
        target_bboxes = list(gt["boxes"] for gt in targets)
        # unscaled
        for i, boxes in enumerate(target_bboxes):
            boxes = box_ops.box_convert(boxes, "cxcywh", "xyxy")
            boxes[:, [0, 2]] *= target_sizes[i][1]
            boxes[:, [1, 3]] *= target_sizes[i][0]
            target_bboxes[i] = boxes
        return target_bboxes

    def __call__(
        self,
        model: nn.Module,
        reset_on_done: bool = True
    ) -> Dict[str, Any]:
        """
        Args:
            reset_on_done: automatically call self.reset() when valuation is done
        Return:
            dictionary:
            {
                loss:
                loss_dict:
                performance:
            }
        """
        model.eval()
        self.criterion.eval()

        # only show in main process
        tqdm_shower = None
        if self.main_process:
            tqdm_shower = tqdm.tqdm(total=len(self.val_loader), desc="Val Batch")
        dist_utils.barrier()
        loss_meter = metrics.AverageMeter()
        loss_dict_meter = metrics.DictAverageMeter()

        with torch.no_grad():
            for samples, targets in self.val_loader:
                samples, targets = move_data_to_device(samples, targets, self.device)
                try:
                    output = model(samples)
                    # calculate loss
                    loss, loss_dict = self.get_loss(output, targets)
                # critical error
                except detr_utils.ErrorBBOX as e:
                    os.makedirs("debug", exist_ok=True)
                    fp = "debug/val_error_bbox_{}.pth".format(dist_utils.get_rank())
                    print(f"bbox is error: {e.bbox}, store in {fp}")
                    state_dict = {
                        "model": model.state_dict(),
                        "bbox": e.bbox,
                        "x": samples.tensors,
                        "mask": samples.mask,
                        "targets": targets
                    }
                    torch.save(state_dict, fp)
                    raise
                # calculate predicted bounding boxes (xyxy w.r.t. size after data augmentation)
                pred_bboxes = self.get_pred_bboxes(output, targets)
                self.update_prediction(pred_bboxes, targets)
                loss_meter.update(loss)
                loss_dict_meter.update(loss_dict)
                if self.main_process:
                    tqdm_shower.update()
                dist_utils.barrier()
        if self.main_process:
            tqdm_shower.close()
        dist_utils.barrier()

        # accumulate
        loss_meter.accumulate()
        loss_dict_meter.accumulate()
        loss_meter.sync()
        loss_dict_meter.sync()

        ret = dict(
            loss=loss_meter.value(),
            loss_dict=loss_dict_meter.value(),
            performance=self.get_performance()
        )
        if reset_on_done:
            self.reset()
        return ret

