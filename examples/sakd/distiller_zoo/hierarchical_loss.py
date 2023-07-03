import json
import logging
from typing import Dict, List

import torch
from torch import nn
from torch.nn import Module, ModuleDict

from .sub_modules import FeatureClassify


class HierarchicalLoss(Module):
    def __init__(self, layer_cluster_info: Dict[str, str], feat_s: List[torch.Tensor]):
        """
        Args:
            layer_cluster_info: cluster info filepath by layer name
        """
        self.logger = logging.getLogger("HierarchicalLoss")

        super().__init__()

        # loading cluster info from merge.json files
        self.cluster_info: Dict[str, List[List[int]]] = dict()
        for name, info_file in layer_cluster_info.items():
            with open(info_file, "r") as f:
                info: List[List[int]] = json.load(f)["merged_classes"]
            self.cluster_info[name] = info

        # for each extracted layer, create classifier
        self.layer_cls_loss_fn: ModuleDict[str, nn.CrossEntropyLoss] = ModuleDict()
        layer_cls_dict: Dict[str, FeatureClassify] = dict()
        for name, info in self.cluster_info.items():
            idx = int(name[-1])
            self.logger.warning(
                "Mapping teacher layer: %s to the %d-th part of student output feature with shape: %s!!!",
                name, idx, feat_s[idx].shape
            )
            layer_cls_dict[name] = FeatureClassify(
                input_channels=feat_s[idx].shape[1],
                num_classes=len(info)
            )
            self.layer_cls_loss_fn[name] = nn.CrossEntropyLoss(weight=self._get_cluster_weight(info))

        self.feature_classifiers = ModuleDict(layer_cls_dict)
        self.layer_cluster_map = self._layer_cluster_class_map()

    def _get_cluster_weight(self, info: List[List[int]]):
        weight = torch.empty(len(info))
        for idx, cluster in enumerate(info):
            num = len(cluster)
            weight[idx] = 1 / num
        return weight

    def _layer_cluster_class_map(self) -> Dict[str, Dict[int, int]]:
        layer_cluster_map = dict()
        for name, info in self.cluster_info.items():
            cluster_map = dict()
            for cluster_id, cluster in enumerate(info):
                for fine_id in cluster:
                    cluster_map[fine_id] = cluster_id
            layer_cluster_map[name] = cluster_map
        return layer_cluster_map

    def cluster_cls_loss(
        self,
        layer_name: str,
        pred: torch.Tensor,
        target: torch.Tensor
    ):
        cluster_map = self.layer_cluster_map[layer_name]
        t = torch.empty_like(target)
        assert target.dim() == 1, "target dim {} is not equal to 1"
        for i, x in enumerate(target):
            t[i] = cluster_map[x.item()]
        loss = self.layer_cls_loss_fn[layer_name](pred, t)
        return loss

    def forward(self, feat_s: List[torch.Tensor], logit_t: torch.Tensor):
        losses: List[torch.Tensor] = list()
        for name, classifier in self.feature_classifiers.items():
            idx = int(name[-1])
            pred = classifier(feat_s[idx])
            losses.append(self.cluster_cls_loss(name, pred, logit_t))
        return sum(losses)




