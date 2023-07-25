from typing import Dict, Any

from .backbone import *
from .position_encoding import *
from .transformer import Transformer
from .detr import DETR
from .updetr import UPDETR
from .seq_dropout import build_seq_dropout


def build_backbone(train_seg: bool, model_cfg: Dict[str, Any]) -> Joiner:
    position_embedding = build_position_encoding(
        hidden_dim=model_cfg["transformer"]["embed_dim"],
        name=model_cfg["position_encoding"]["name"]
    )
    backbone = Backbone(return_interm_layers=train_seg, **model_cfg["backbone"])
    joiner = Joiner(backbone, position_embedding)
    return joiner


def build_transformer(transformer_cfg: Dict[str, Any]) -> Transformer:
    trans = Transformer(**transformer_cfg)
    return trans


def build_detr(model_cfg: Dict[str, Any], num_classes: int, seg: bool = False) -> DETR:
    """
    Build DETR module
    Args:
        seg: if `True`, enable segmentation
    """
    backbone = build_backbone(seg, model_cfg)
    transformer = build_transformer(model_cfg["transformer"])
    seq_dropout = build_seq_dropout(model_cfg)
    detr_cfg = model_cfg["detr"]
    detr = DETR(
        backbone=backbone,
        transformer=transformer,
        num_classes=num_classes,
        seq_dropout=seq_dropout,
        **detr_cfg
    )
    return detr


def build_updetr(model_cfg: Dict[str, Any], num_classes: int = 2) -> UPDETR:
    """
    Build UPDETR module
    """
    backbone = build_backbone(False, model_cfg)
    transformer = build_transformer(model_cfg["transformer"])
    updetr_cfg = model_cfg["updetr"]
    detr = UPDETR(
        backbone=backbone,
        transformer=transformer,
        num_classes=num_classes,
        **updetr_cfg
    )
    return detr
