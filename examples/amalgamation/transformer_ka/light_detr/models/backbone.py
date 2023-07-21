"""
Backbone modules.
"""
from functools import partial
from typing import Dict, List, OrderedDict

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from light_detr.utils.nested_tensor import NestedTensor
from .position_encoding import PositionEmbeddingBase


__all__ = [
    "Backbone",
    "Joiner"
]


class BackboneBase(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        train_backbone: bool,
        num_channels: int,
        return_interm_layers: bool = False
    ):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or "layer2" not in name and "layer3" not in name and "layer4" not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def patch_forward(self, patch: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        """
        For unsupervised pretrain
        """
        return self.body(patch)

    def forward(self, padded_x: NestedTensor):
        xs = self.body(padded_x.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = padded_x.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(
        self,
        backbone_name: str,
        train_backbone: bool,
        return_interm_layers: bool,
        dilation: bool
    ):
        backbone = getattr(torchvision.models, backbone_name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=True,
            norm_layer=partial(FrozenBatchNorm2d, eps=1e-5)
        )
        # load the SwAV pre-training model from the url instead of supervised pre-training model
        if backbone_name == "resnet50":
            checkpoint = torch.hub.load_state_dict_from_url(
                "https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar",
                map_location="cpu"
            )
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
            backbone.load_state_dict(state_dict, strict=False)
        num_channels = 512 if backbone_name in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Module):
    def __init__(self, backbone: Backbone, position_embedding: PositionEmbeddingBase):
        super().__init__()
        self.backbone = backbone
        self.position_embedding = position_embedding
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.backbone(tensor_list)
        out: List[NestedTensor] = []
        pos: List[torch.Tensor] = []
        for _, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self.position_embedding(x).to(x.tensors.dtype))

        return out, pos

    def patch_forward(self, patch: torch.Tensor):
        out_dict = self.backbone.patch_forward(patch)
        return list(out_dict.values())

