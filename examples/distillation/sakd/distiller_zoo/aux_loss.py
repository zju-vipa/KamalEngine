from typing import Dict, Any, List

import torch
from torch.nn import Module, ModuleDict, MSELoss, ModuleList
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from helper.pretrain import init_pretrain

from models.util import Embed, ConvReg, LinearEmbed
from models.util import Connector, Translator, Paraphraser

from distiller_zoo.AB import ABLoss
from distiller_zoo.CC import Correlation
from distiller_zoo.FitNet import HintLoss
from distiller_zoo.FSP import FSP
from distiller_zoo.FT import FactorTransfer
from distiller_zoo.VID import VIDLoss
# from distiller_zoo.crd import CRDLoss
from distiller_zoo.hierarchical_loss import HierarchicalLoss
from distiller_zoo.mid_cls_loss import MidClsLoss


def get_sample_feat(cfg: Dict[str, Any], module_dict: ModuleDict, device: torch.device):
    """
    Return sample featuremap of teacher and student
    """
    data = torch.randn(2, 3, cfg["dataset"]["img_h"], cfg["dataset"]["img_w"]).to(device)
    module_dict.eval()
    feat_s, _ = module_dict["student"](data, is_feat=True)
    feat_t, _ = module_dict["teacher"](data, is_feat=True)

    return feat_s, feat_t


def ab_loss(
    cfg: Dict[str, Any],
    module_dict: ModuleDict,
    train_loader: DataLoader,
    tb_writer: SummaryWriter,
    device: torch.device,
    **kwargs
):
    feat_s, feat_t = get_sample_feat(cfg, module_dict, device)
    s_shapes = [f.shape for f in feat_s[1:-1]]
    t_shapes = [f.shape for f in feat_t[1:-1]]
    connector = Connector(s_shapes, t_shapes).to(device)
    # init stage training
    init_trainable_dict = ModuleDict(
        dict(
            connector=connector,
            feature_modules=module_dict["student"].get_feat_modules()
        )
    )
    criterion_kd = ABLoss(len(feat_s[1:-1])).to(device)
    init_pretrain(
        cfg=cfg,
        module_dict=module_dict,
        init_modules=init_trainable_dict,
        criterion=criterion_kd,
        train_loader=train_loader,
        tb_writer=tb_writer,
        device=device
    )
    # classification
    module_dict["connector"] = connector
    return criterion_kd, init_trainable_dict


def correlation_loss(
    cfg: Dict[str, Any],
    module_dict: ModuleDict,
    device: torch.device,
    **kwargs
):
    feat_s, feat_t = get_sample_feat(cfg, module_dict, device)

    criterion_kd = Correlation().to(device)

    embed_s = LinearEmbed(feat_s[-1].shape[1], cfg["kd_loss"]["feat_dim"]).to(device)
    embed_t = LinearEmbed(feat_t[-1].shape[1], cfg["kd_loss"]["feat_dim"]).to(device)
    module_dict["embed_s"] = embed_s
    module_dict["embed_t"] = embed_t

    trainable_dict = ModuleDict(
        dict(
            embed_s=embed_s,
            embed_t=embed_t
        )
    )
    return criterion_kd, trainable_dict


def hint_loss(
    cfg: Dict[str, Any],
    module_dict: ModuleDict,
    device: torch.device,
    **kwargs
):
    feat_s, feat_t = get_sample_feat(cfg, module_dict, device)
    hint_layer = cfg["kd_loss"]["hint_layer"]
    regress_s = ConvReg(feat_s[hint_layer].shape, feat_t[hint_layer].shape)

    criterion_kd = HintLoss(conv_reg=regress_s, hint_layer=hint_layer).to(device)

    module_dict["conv_reg"] = regress_s
    trainable_dict = ModuleDict(
        dict(conv_reg=regress_s)
    )
    return criterion_kd, trainable_dict


def fsp_loss(
    cfg: Dict[str, Any],
    module_dict: ModuleDict,
    train_loader: DataLoader,
    tb_writer: SummaryWriter,
    device: torch.device,
    **kwargs
):
    feat_s, feat_t = get_sample_feat(cfg, module_dict, device)
    s_shapes = [s.shape for s in feat_s[:-1]]
    t_shapes = [t.shape for t in feat_t[:-1]]

    criterion_kd = FSP(s_shapes, t_shapes).to(device)

    # init stage training
    init_trainable_dict = ModuleDict(
        dict(
            student_feat_modules=module_dict["student"].get_feat_modules()
        )
    )
    init_pretrain(
        cfg=cfg,
        module_dict=module_dict,
        init_modules=init_trainable_dict,
        criterion=criterion_kd,
        train_loader=train_loader,
        tb_writer=tb_writer,
        device=device
    )
    return criterion_kd, ModuleDict()


def factor_transfer(
    cfg: Dict[str, Any],
    module_dict: ModuleDict,
    train_loader: DataLoader,
    tb_writer: SummaryWriter,
    device: torch.device,
    **kwargs
):
    feat_s, feat_t = get_sample_feat(cfg, module_dict, device)
    s_shape = feat_s[-2].shape
    t_shape = feat_t[-2].shape

    paraphraser = Paraphraser(t_shape).to(device)
    translator = Translator(s_shape, t_shape).to(device)
    # init stage training
    init_trainable_dict = ModuleDict(
        dict(
            paraphraser=paraphraser
        )
    )
    criterion_init = MSELoss()
    init_pretrain(
        cfg=cfg,
        module_dict=module_dict,
        init_modules=init_trainable_dict,
        criterion=criterion_init,
        train_loader=train_loader,
        tb_writer=tb_writer,
        device=device
    )
    # classification
    criterion_kd = FactorTransfer(
        p1=cfg["kd_loss"]["p1"],
        p2=cfg["kd_loss"]["p2"]
    ).to(device)

    module_dict["translator"] = translator
    module_dict["paraphraser"] = paraphraser
    trainable_dict = ModuleDict(
        dict(translator=translator)
    )
    return criterion_kd, trainable_dict


def vid_loss(
    cfg: Dict[str, Any],
    module_dict: ModuleDict,
    device: torch.device,
    **kwargs
):
    feat_s, feat_t = get_sample_feat(cfg, module_dict, device)
    s_n = [f.shape[1] for f in feat_s[1:-1]]
    t_n = [f.shape[1] for f in feat_t[1:-1]]

    criterion_kd = ModuleList(
        [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
    ).to(device)

    # add this as some parameters in VIDLoss need to be updated
    trainable_dict = ModuleDict(
        dict(criterion_kd_list=criterion_kd)
    )
    return criterion_kd, trainable_dict


def get_hierarchical_loss(
    cfg: Dict[str, Any],
    module_dict: ModuleDict,
    device: torch.device,
    **kwargs
):
    # extracting features
    feat_s, feat_t = get_sample_feat(cfg, module_dict, device)
    criterion_kd = HierarchicalLoss(cfg["kd_loss"]["layer_cluster_info"], feat_s).to(device)
    return criterion_kd, criterion_kd.feature_classifiers


def get_mid_cls_loss(
    cfg: Dict[str, Any],
    module_dict: ModuleDict,
    device: torch.device,
    **kwargs
):
    # extracting features
    num_classes = len(kwargs["train_loader"].dataset.classes)
    feat_s, feat_t = get_sample_feat(cfg, module_dict, device)
    criterion_kd = MidClsLoss(
        num_classes=num_classes,
        mid_layer_T=cfg["kd_loss"]["mid_layer_T"],
        feat_s=feat_s
    ).to(device)
    return criterion_kd, criterion_kd.layer_classifier


AUX_LOSS_MAP = dict(
    ABLoss=ab_loss,
    Correlation=correlation_loss,
    HintLoss=hint_loss,
    FSP=fsp_loss,
    FactorTransfer=factor_transfer,
    VIDLoss=vid_loss,
    # CRDLoss=crd_loss,
    HierarchicalLoss=get_hierarchical_loss,
    MidClsLoss=get_mid_cls_loss
)


def get_aux_loss_modules(cfg: Dict[str, Any], **kwargs):
    return AUX_LOSS_MAP[cfg["kd_loss"]["name"]](cfg, **kwargs)
