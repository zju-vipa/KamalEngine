from . import functional, loss
from .aux_loss import get_aux_loss_modules
from .loss import *

from typing import Dict, Any
import copy

import torch
from torch.nn import ModuleDict, CrossEntropyLoss, ModuleList
from torch import Tensor

from .hierarchical_loss import HierarchicalLoss


def base_kd_forward(**kwargs):
    device = kwargs["target"].device
    return torch.tensor(0.0, device=device, dtype=torch.float)


def hint_forward(**kwargs):
    feat_s = kwargs["feat_s"]
    feat_t = kwargs["feat_t"]

    criterion_kd: HintLoss = kwargs["criterion_kd"]
    return criterion_kd(feat_s, feat_t)


def nst_forward(**kwargs):
    feat_s = kwargs["feat_s"]
    feat_t = kwargs["feat_t"]
    g_s = feat_s[1:-1]
    g_t = feat_t[1:-1]
    criterion_kd: NSTLoss = kwargs["criterion_kd"]
    loss_group = criterion_kd(g_s, g_t)
    loss_kd = sum(loss_group)
    return loss_kd

def rkd_forward(**kwargs):
    feat_s = kwargs["feat_s"]
    feat_t = kwargs["feat_t"]
    f_s = feat_s[-1]
    f_t = feat_t[-1]

    criterion_kd: RKDLoss = kwargs["criterion_kd"]
    loss_kd = criterion_kd(f_s, f_t)
    return loss_kd

def kdsvd_forward(**kwargs):
    pass
#             g_s = feat_s[1:-1]
#             g_t = feat_t[1:-1]
#             loss_group = criterion_kd(g_s, g_t)
#             loss_kd = sum(loss_group)


def correlation_forward(**kwargs):
    feat_s = kwargs["feat_s"]
    feat_t = kwargs["feat_t"]

    f_s = kwargs["module_dict"]["embed_s"](feat_s[-1])
    f_t = kwargs["module_dict"]["embed_t"](feat_t[-1])
    criterion_kd: Correlation = kwargs["criterion_kd"]
    loss_kd = criterion_kd(f_s, f_t)
    return loss_kd


def vid_forward(**kwargs):
    feat_s = kwargs["feat_s"]
    feat_t = kwargs["feat_t"]

    g_s = feat_s[1:-1]
    g_t = feat_t[1:-1]
    criterion_kd: ModuleList[VIDLoss] = kwargs["criterion_kd"]
    loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
    loss_kd = sum(loss_group)
    return loss_kd

def hierarchical_forward(**kwargs):
    feat_s = kwargs["feat_s"]
    target = kwargs["target"]
    criterion_kd: HierarchicalLoss = kwargs["criterion_kd"]
    return criterion_kd(feat_s, target)

def mid_cls_forward(**kwargs):
    feat_s = kwargs["feat_s"]
    target = kwargs["target"]
    criterion_kd: MidClsLoss = kwargs["criterion_kd"]
    return criterion_kd(feat_s, target)


KD_LOSS_DICT = dict(
    DistillKL=DistillKL,
    NSTLoss=NSTLoss,
    RKDLoss=RKDLoss,
)


AUX_KD_LOSS_DICT = dict(
    ABLoss=ABLoss,
    Correlation=Correlation,
    HintLoss=HintLoss,
    VIDLoss=VIDLoss,
    HierarchicalLoss=HierarchicalLoss,
    MidClsLoss=MidClsLoss
)


KD_LOSS_FORWARD_DICT = dict(
    DistillKL=base_kd_forward,
    KDSVD=kdsvd_forward,
    NSTLoss=nst_forward,
    RKDLoss=rkd_forward,
    ABLoss=base_kd_forward,
    Correlation=correlation_forward,
    HintLoss=hint_forward,
    FSP=base_kd_forward,
    VIDLoss=vid_forward,
    HierarchicalLoss=hierarchical_forward,
    MidClsLoss=mid_cls_forward
)


def get_loss_module(cfg: Dict[str, Any], device: torch.device, **kwargs) -> ModuleDict:
    """
    kwargs:
        module_dict: ModuleDict,
        train_loader: DataLoader,
        tb_writer: SummaryWriter,
        device: torch.device
    """
    criterion_cls = CrossEntropyLoss().to(device)
    criterion_div = DistillKL(cfg["kd_loss"]["KD_T"]).to(device)

    loss_cfg = copy.deepcopy(cfg["kd_loss"])
    loss_cfg.pop("KD_T")
    name = loss_cfg["name"]
    if name in KD_LOSS_DICT.keys():
        loss_module = KD_LOSS_DICT[name]
        loss_cfg.pop("name")
        criterion_kd = loss_module(**loss_cfg).to(device)
        trainable_dict = ModuleDict()
    elif name in AUX_KD_LOSS_DICT.keys():
        criterion_kd, trainable_dict = get_aux_loss_modules(cfg, device=device, **kwargs)
    else:
        raise Exception("Loss: {} not supported".format(name))

    criterion_dict = ModuleDict(dict(
        cls=criterion_cls,
        div=criterion_div,
        kd=criterion_kd
    ))
    return criterion_dict, trainable_dict


def get_loss_forward(cfg, **kwargs) -> Tensor:
    """
    kwargs:
        feat_s,
        feat_t,
        logit_s,
        logit_t,
        target,
        criterion_kd,
        module_dict
    """
    return KD_LOSS_FORWARD_DICT[cfg["kd_loss"]["name"]](**kwargs)

