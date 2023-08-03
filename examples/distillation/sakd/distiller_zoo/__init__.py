from typing import Dict, Any
import copy

import torch
from torch.nn import ModuleDict, CrossEntropyLoss, ModuleList, NLLLoss, KLDivLoss
from torch import Tensor

from .AB import ABLoss
from .AT import Attention
from .CC import Correlation
from .FitNet import HintLoss
from .FSP import FSP
from .FT import FactorTransfer
from .KD import DistillKL, DistillKL_NLL
from .KDSVD import KDSVD
from .NST import NSTLoss
from .PKT import PKT
from .RKD import RKDLoss
from .SP import Similarity
from .VID import VIDLoss
from .hierarchical_loss import HierarchicalLoss
from .mid_cls_loss import MidClsLoss

from .aux_loss import get_aux_loss_modules


def base_kd_forward(**kwargs):
    device = kwargs["target"].device
    return torch.tensor(0.0, device=device, dtype=torch.float)


def hint_forward(**kwargs):
    feat_s = kwargs["feat_s"]
    feat_t = kwargs["feat_t"]

    criterion_kd: HintLoss = kwargs["criterion_kd"]
    return criterion_kd(feat_s, feat_t)


def attention_forward(**kwargs):
    feat_s = kwargs["feat_s"]
    feat_t = kwargs["feat_t"]
    g_s = feat_s[1:-1]
    g_t = feat_t[1:-1]
    criterion_kd: Attention = kwargs["criterion_kd"]
    loss_group = criterion_kd(g_s, g_t)
    loss_kd = sum(loss_group)
    return loss_kd


def nst_forward(**kwargs):
    feat_s = kwargs["feat_s"]
    feat_t = kwargs["feat_t"]
    g_s = feat_s[1:-1]
    g_t = feat_t[1:-1]
    criterion_kd: NSTLoss = kwargs["criterion_kd"]
    loss_group = criterion_kd(g_s, g_t)
    loss_kd = sum(loss_group)
    return loss_kd


def similarity_forward(**kwargs):
    feat_s = kwargs["feat_s"]
    feat_t = kwargs["feat_t"]

    g_s = [feat_s[-2]]
    g_t = [feat_t[-2]]
    criterion_kd: Similarity = kwargs["criterion_kd"]
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


def pkt_forward(**kwargs):
    feat_s = kwargs["feat_s"]
    feat_t = kwargs["feat_t"]
    f_s = feat_s[-1]
    f_t = feat_t[-1]
    criterion_kd: PKT = kwargs["criterion_kd"]
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


def factor_forward(**kwargs):
    feat_s = kwargs["feat_s"]
    feat_t = kwargs["feat_t"]

    factor_s = kwargs["module_dict"]["translator"](feat_s[-2])
    factor_t = kwargs["module_dict"]["paraphraser"](feat_t[-2], is_factor=True)
    criterion_kd: FactorTransfer = kwargs["criterion_kd"]
    loss_kd = criterion_kd(factor_s, factor_t)
    return loss_kd


def hierarchical_forward(**kwargs):
    feat_s = kwargs["feat_s"]
    target = kwargs["target"]
    criterion_kd: MidClsLoss = kwargs["criterion_kd"]
    return criterion_kd(feat_s, target)


def mid_cls_forward(**kwargs):
    feat_s = kwargs["feat_s"]
    logit_t = kwargs["logit_t"]
    criterion_kd: HierarchicalLoss = kwargs["criterion_kd"]
    return criterion_kd(feat_s, logit_t)


KD_LOSS_DICT = dict(
    Attention=Attention,
    DistillKL=DistillKL,
    DistillKL_NLL=DistillKL_NLL,
    KDSVD=KDSVD,
    NSTLoss=NSTLoss,
    PKT=PKT,
    RKDLoss=RKDLoss,
    Similarity=Similarity,
)


AUX_KD_LOSS_DICT = dict(
    ABLoss=ABLoss,
    Correlation=Correlation,
    HintLoss=HintLoss,
    FSP=FSP,
    FactorTransfer=FactorTransfer,
    VIDLoss=VIDLoss,
    HierarchicalLoss=HierarchicalLoss,
    MidClsLoss=MidClsLoss
)


KD_LOSS_FORWARD_DICT = dict(
    Attention=attention_forward,
    DistillKL=base_kd_forward,
    CRD=base_kd_forward,
    DistillKL_NLL=base_kd_forward,
    KDSVD=kdsvd_forward,
    NSTLoss=nst_forward,
    PKT=pkt_forward,
    RKDLoss=rkd_forward,
    Similarity=similarity_forward,
    ABLoss=base_kd_forward,
    Correlation=correlation_forward,
    HintLoss=hint_forward,
    FSP=base_kd_forward,
    FactorTransfer=factor_forward,
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
    # if name in KD_LOSS_DICT.keys():
    #     loss_module = KD_LOSS_DICT[name]
    #     loss_cfg.pop("name")
    #     criterion_kd = loss_module(**loss_cfg).to(device)
    #     trainable_dict = ModuleDict()
    # elif name in AUX_KD_LOSS_DICT.keys():
    #     criterion_kd, trainable_dict = get_aux_loss_modules(cfg, device=device, **kwargs)
    # # elif name in ['CRD','AT']:
    # #     criterion_kd=CrossEntropyLoss().to(device)
    # #     trainable_dict = ModuleDict()
    # else:
    criterion_kd = CrossEntropyLoss().to(device)
    trainable_dict = ModuleDict()
    #raise Exception("Loss: {} not supported".format(name))

    criterion_dict = ModuleDict(dict(
        cls=criterion_cls,
        div=criterion_div,
        kd=criterion_kd,
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

