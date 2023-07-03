import logging
from typing import Dict, List

import torch
from torch.nn import Module, ModuleDict

from .sub_modules import FeatureClassify
from .KD import DistillKL


class MidClsLoss(Module):
    def __init__(self, num_classes: int, mid_layer_T: Dict[str, float], feat_s: List[torch.Tensor]):
        self.logger = logging.getLogger("HierarchicalLoss")
        super().__init__()

        self.layer_classifier: ModuleDict[str, FeatureClassify] = ModuleDict()
        self.layer_loss: ModuleDict[str, DistillKL] = ModuleDict()
        for name, T in mid_layer_T.items():
            idx = int(name[-1])
            self.logger.warning(
                "Mapping teacher layer: %s to the %d-th part of student output feature with shape: %s!!!",
                name, idx, feat_s[idx].shape
            )
            self.layer_classifier[name] = FeatureClassify(
                input_channels=feat_s[idx].shape[1],
                num_classes=num_classes
            )
            self.layer_loss[name] = DistillKL(T)

    def forward(self, feat_s: List[torch.Tensor], target: torch.Tensor):
        losses: List[torch.Tensor] = list()
        for name in self.layer_classifier.keys():
            idx = int(name[-1])
            pred = self.layer_classifier[name](feat_s[idx])
            losses.append(self.layer_loss[name](pred, target))
        return sum(losses)

