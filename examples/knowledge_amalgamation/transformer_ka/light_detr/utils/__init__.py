from typing import List, Any, Dict, Tuple

import torch
from torch import nn

from .nested_tensor import *
from .dist_utils import LogArgs, DistLaunchArgs


def move_data_to_device(
    x: NestedTensor,
    targets: List[Dict[str, Any]],
    device: torch.device
) -> Tuple[NestedTensor, List[Dict[str, Any]]]:
    x = x.to(device)
    for i, target in enumerate(targets):
        for k, v in target.items():
            if isinstance(v, torch.Tensor):
                target[k] = v.to(device)
        targets[i] = target
    return x, targets


class ErrorNan(Exception):
    def __init__(self, nan_tensor: torch.Tensor, *args: object) -> None:
        super().__init__(*args)
        self.nan_tensor = nan_tensor


class ErrorBBOX(Exception):
    def __init__(self, bbox: torch.Tensor, *args: object) -> None:
        super().__init__(*args)
        self.bbox = bbox


def load_pretrain_model(fp: str, model: nn.Module, num_proj: int = 1):
    ckpt = torch.load(fp, map_location="cpu")
    pretrain_weights = ckpt["model"]
    for k in list(pretrain_weights.keys()):
        if "class_embed" in k:
            pretrain_weights.pop(k)
    if num_proj > 1:
        proj_weight = pretrain_weights["input_proj.weight"]
        proj_bias = pretrain_weights["input_proj.bias"]
        pretrain_weights["input_proj.weight"] = proj_weight.repeat(num_proj, 1, 1, 1)
        pretrain_weights["input_proj.bias"] = proj_bias.repeat(num_proj)
    model.load_state_dict(pretrain_weights, strict=False)
