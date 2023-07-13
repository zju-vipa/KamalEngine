# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
from typing import Dict, List, Any, Tuple
from collections import OrderedDict

import torch
from torch import Tensor
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.ops.boxes as box_ops

from .matcher import HungarianMatcher

from cv_lib.distributed.utils import is_dist_avail_and_initialized, get_world_size
from cv_lib.metrics.cls_acc import accuracy


class SetCriterion(nn.Module):
    """
    This class computes the loss for DETR. The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        eos_coef: float,
        loss_items: List[str],
        loss_name_prefix: str = "",
        kd_temp: float = 4.0
    ):
        """
        Create the criterion.

        Args:
            num_classes: number of object categories, including the special no-object category
            matcher: module able to compute a matching between targets and proposals
            eos_coef: relative classification weight applied to the no-object category
            loss_items: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        # removed background class
        self.num_categories = num_classes - 1
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.loss_items = loss_items
        self.loss_name_prefix = loss_name_prefix
        self.kd_T = kd_temp
        empty_weight = torch.ones(self.num_classes)
        # no-object index is 0
        empty_weight[0] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(
        self,
        outputs: Dict[str, Tensor],
        targets: List[dict],
        indices: List[Tuple[Tensor, Tensor]],
        record_class_error=True,
        **kwargs
    ):
        """
        Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros(
            src_logits.shape[:2],
            dtype=torch.long,
            device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}

        if record_class_error:
            losses["class_error"] = 100 - 100 * accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_kd(
        self,
        outputs: Dict[str, Tensor],
        targets: List[dict],
        indices: List[Tuple[Tensor, Tensor]],
        **kwargs
    ):
        """
        KD loss (KL divergence)
        targets dicts must contain the key "logits_t" containing a tensor of dim [nb_target_boxes]
        """
        if "logits_t" not in targets[0]:
            return dict()

        # NIL loss for unmatched predictions
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]
        device = src_logits.device

        idx = self._get_src_permutation_idx(indices)
        unmatched_mask = torch.ones(src_logits.shape[:2], dtype=torch.bool, device=device)
        unmatched_mask[idx] &= False
        unmatched_logits = src_logits[unmatched_mask]
        target_classes = torch.zeros(unmatched_logits.shape[0], dtype=torch.long, device=device)
        loss_ce = F.cross_entropy(unmatched_logits, target_classes, self.empty_weight, reduction="sum")

        # KL divergence for matched preditions
        target_soft_o = torch.cat([t["logits_t"][J] for t, (_, J) in zip(targets, indices)])
        # kd loss
        p_s = F.log_softmax(src_logits[idx] / self.kd_T, dim=1)
        p_t = F.softmax(target_soft_o / self.kd_T, dim=1)
        loss_kl = F.kl_div(p_s, p_t, reduction="sum") * (self.kd_T ** 2)
        loss_kd = (loss_ce + loss_kl) / (target_soft_o.shape[0] + self.empty_weight[0] * target_classes.shape[0])
        # total losses
        losses = {"loss_kd": loss_kd}

        return losses

    def loss_feature(
        self,
        outputs: Dict[str, Tensor],
        indices: List[Tuple[Tensor, Tensor]],
        num_boxes: float,
        **kwargs
    ):
        """
        Compute the mse loss between normalized features.
        """
        target_feature = outputs['gt_feature']
        idx = self._get_src_permutation_idx(indices)
        batch_size = len(indices)
        target_feature = target_feature.view(batch_size, target_feature.shape[0] // batch_size, -1)

        src_feature = outputs['pred_feature'][idx]
        target_feature = torch.cat([t[i] for t, (_, i) in zip(target_feature, indices)], dim=0)

        # l2 normalize the feature
        src_feature = F.normalize(src_feature, dim=1)
        target_feature = F.normalize(target_feature, dim=1)

        loss_feature = F.mse_loss(src_feature, target_feature, reduction='none')
        losses = {'loss_feature': loss_feature.sum() / num_boxes}

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs: Dict[str, Tensor], targets: List[dict], **kwargs):
        """
        Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is 0)
        card_pred = (pred_logits.argmax(-1) != 0).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(
        self,
        outputs: Dict[str, Tensor],
        targets: List[dict],
        indices: List[Tuple[Tensor, Tensor]],
        num_boxes: float,
        **kwargs
    ):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_convert(src_boxes, "cxcywh", "xyxy"),
            box_ops.box_convert(target_boxes, "cxcywh", "xyxy")
        ))
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    @staticmethod
    def _get_src_permutation_idx(indices: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @staticmethod
    def _get_tgt_permutation_idx(indices: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(
        self,
        loss_name: str,
        outputs: Dict[str, Tensor],
        targets: List[dict],
        **kwargs
    ) -> Dict[str, Tensor]:
        loss_map = {
            "labels": self.loss_labels,
            "kd": self.loss_kd,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "feature": self.loss_feature
        }
        return loss_map[loss_name](outputs=outputs, targets=targets, **kwargs)

    def forward(self, outputs: Dict[str, Any], targets: List[dict]) -> Dict[str, Tensor]:
        """
        This performs the loss computation.
        Args:
            outputs: see the output specification of the model for the format
            targets: list of dicts, such that len(targets) == batch_size.
                The expected keys in each dict depends on the losses applied, see each loss" doc
        """
        # shallow copy
        aux_outputs = outputs.copy().pop("aux_outputs", list())
        device = list(outputs.values())[0].device

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.tensor(num_boxes, dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            dist.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses: Dict[str, Tensor] = OrderedDict()
        for loss_name in self.loss_items:
            loss_item = self.get_loss(
                loss_name,
                outputs,
                targets,
                indices=indices,
                num_boxes=num_boxes
            )
            losses.update(loss_item)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        for layer_id, aux_outputs in enumerate(aux_outputs):
            indices = self.matcher(aux_outputs, targets)
            for loss_name in self.loss_items:
                if loss_name == "masks":
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                l_dict = self.get_loss(
                    loss_name,
                    aux_outputs,
                    targets,
                    indices=indices,
                    num_boxes=num_boxes,
                    record_class_error=False
                )
                l_dict = {k + f".{layer_id}": v for k, v in l_dict.items()}
                losses.update(l_dict)
        # add prefix
        losses = {self.loss_name_prefix + k: v for k, v in losses.items()}
        return losses
