from typing import Dict

import torch
from torch.utils.data import DataLoader

from .evaluation import EvaluationBase
from .eval_voc import EvaluationVOC
from .eval_coco import EvaluationCOCO
from light_detr.loss import SetCriterion


__SUPPORTED_VAL__: Dict[str, EvaluationBase] = {
    "VOC2007": EvaluationVOC,
    "VOC2012": EvaluationVOC,
    "COCO": EvaluationCOCO,
    "COCO_large": EvaluationCOCO,
}


def get_evaluator(
    dataset_name: str,
    criterion: SetCriterion,
    val_loader: DataLoader,
    loss_weights: Dict[str, float],
    device: torch.device
) -> EvaluationBase:
    return __SUPPORTED_VAL__[dataset_name](criterion, val_loader, loss_weights, device)
