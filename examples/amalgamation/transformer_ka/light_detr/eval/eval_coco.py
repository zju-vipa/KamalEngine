from typing import Any, Dict, List

import torch
from torch import Tensor
from torch.utils.data import DataLoader

import cv_lib.metrics as metrics

from .evaluation import EvaluationBase
from light_detr.loss import SetCriterion


class EvaluationCOCO(EvaluationBase):
    """
    Distributed COCO evaluator
    """
    def __init__(
        self,
        criterion: SetCriterion,
        val_loader: DataLoader,
        loss_weights: Dict[str, float],
        device: torch.device
    ):
        super().__init__(criterion, val_loader, loss_weights, device)
        self.metric = metrics.APMeter_COCO(criterion.num_classes)

    def update_prediction(
        self,
        outputs: List[Dict[str, Tensor]],
        targets: List[Dict[str, Any]]
    ):
        pred_bboxes = list(pred["boxes"] for pred in outputs)
        pred_labels = list(pred["labels"] for pred in outputs)
        pred_scores = list(pred["scores"] for pred in outputs)

        gt_bboxes = self.decode_gt_box(targets)
        gt_labels = list(gt["labels"] for gt in targets)
        gt_crowd = list(gt["iscrowd"] for gt in targets)

        img_ids = list(torch.tensor(int(gt["image_id"])) for gt in targets)

        self.metric.update(
            img_ids,
            pred_bboxes,
            pred_labels,
            pred_scores,
            gt_bboxes,
            gt_labels,
            gt_crowd
        )
