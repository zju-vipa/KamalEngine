from typing import Dict, Any, List
import math
import torch
from torch.nn import Module, ModuleDict, MSELoss, ModuleList
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import init_pretrain

from kamal.core.tasks.loss.loss import *
from kamal.slim.distillation.fsp import *
from kamal.slim.distillation.ft import *

def abound_init_loss_forward(
    init_modules: ModuleDict,
    criterion: Module,
    feat_s: List[Tensor],
    feat_t: List[Tensor]
) -> Tensor:
    g_s = init_modules["connector"](feat_s[1:-1])
    g_t = feat_t[1:-1]
    loss_group = criterion(g_s, g_t)
    loss = sum(loss_group)
    return loss


def factor_init_loss_forward(
    init_modules: ModuleDict,
    criterion: Module,
    feat_s: List[Tensor],
    feat_t: List[Tensor]
) -> Tensor:
    f_t = feat_t[-2]
    _, f_t_rec = init_modules["paraphraser"](f_t)
    loss = criterion(f_t_rec, f_t)
    return loss


def fsp_init_loss_forward(
    init_modules: ModuleDict,
    criterion: Module,
    feat_s: List[Tensor],
    feat_t: List[Tensor]
) -> Tensor:
    loss_group = criterion(feat_s[:-1], feat_t[:-1])
    loss = sum(loss_group)
    return loss


INIT_LOSS_FORWARD_DICT = dict(
    ABLoss=abound_init_loss_forward,
    FactorTransfer=factor_init_loss_forward,
    FSP=fsp_init_loss_forward
)


def init_pretrain(
    cfg: Dict[str, Any],
    module_dict: ModuleDict,
    init_modules: ModuleDict,
    criterion: Module,
    train_loader: DataLoader,
    tb_writer: SummaryWriter,
    device: torch.device
):
    logger = logging.getLogger("init_pretrain")
    logger.info("Start pretraining...")

    module_dict.eval()
    init_modules.train()

    optimizer = get_optimizer(
        params=init_modules.parameters(),
        optim_cfg=cfg["pretrain"]["optimizer"]
    )

    losses = AverageMeter()
    for epoch in range(1, cfg["pretrain"]["epochs"] + 1):
        losses.reset()
        for x, target in tqdm.tqdm(train_loader):
            x = x.to(device)
            target = target.to(device)

            # ============= forward ==============
            preact = False
            if cfg["kd_loss"]["name"] == "ABLoss":
                preact = True

            feat_s, _ = module_dict["student"](x, is_feat=True, preact=preact)
            with torch.no_grad():
                feat_t, _ = module_dict["teacher"](x, is_feat=True, preact=preact)
                # feat_t = [f.detach() for f in feat_t]

            loss = INIT_LOSS_FORWARD_DICT[cfg["kd_loss"]["name"]](
                init_modules=init_modules,
                criterion=criterion,
                feat_s=feat_s,
                feat_t=feat_t
            )

            losses.update(loss.item(), x.shape[0])

            # ===================backward=====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # end of epoch
        tb_writer.add_scalar('epoch/init/loss', losses.avg, epoch)
        logger.info("Epoch: [%3d|%3d], loss: %.5f", epoch, cfg["pretrain"]["epochs"], losses.val)

class Paraphraser(nn.Module):
    """Paraphrasing Complex Network: Network Compression via Factor Transfer"""
    def __init__(self, t_shape, k=0.5, use_bn=False):
        super(Paraphraser, self).__init__()
        in_channel = t_shape[1]
        out_channel = int(t_shape[1] * k)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(out_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, f_s, is_factor=False):
        factor = self.encoder(f_s)
        if is_factor:
            return factor
        rec = self.decoder(factor)
        return factor, rec


class Translator(nn.Module):
    def __init__(self, s_shape, t_shape, k=0.5, use_bn=True):
        super(Translator, self).__init__()
        in_channel = s_shape[1]
        out_channel = int(t_shape[1] * k)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, f_s):
        return self.encoder(f_s)


class Connector(nn.Module):
    """Connect for Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons"""
    def __init__(self, s_shapes, t_shapes):
        super(Connector, self).__init__()
        self.s_shapes = s_shapes
        self.t_shapes = t_shapes

        self.connectors = nn.ModuleList(self._make_conenctors(s_shapes, t_shapes))

    @staticmethod
    def _make_conenctors(s_shapes, t_shapes):
        assert len(s_shapes) == len(t_shapes), 'unequal length of feat list'
        connectors = []
        for s, t in zip(s_shapes, t_shapes):
            if s[1] == t[1] and s[2] == t[2]:
                connectors.append(nn.Sequential())
            else:
                connectors.append(ConvReg(s, t, use_relu=False))
        return connectors

    def forward(self, g_s):
        out = []
        for i in range(len(g_s)):
            out.append(self.connectors[i](g_s[i]))

        return out


class ConnectorV2(nn.Module):
    """A Comprehensive Overhaul of Feature Distillation (ICCV 2019)"""
    def __init__(self, s_shapes, t_shapes):
        super(ConnectorV2, self).__init__()
        self.s_shapes = s_shapes
        self.t_shapes = t_shapes

        self.connectors = nn.ModuleList(self._make_conenctors(s_shapes, t_shapes))

    def _make_conenctors(self, s_shapes, t_shapes):
        assert len(s_shapes) == len(t_shapes), 'unequal length of feat list'
        t_channels = [t[1] for t in t_shapes]
        s_channels = [s[1] for s in s_shapes]
        connectors = nn.ModuleList([self._build_feature_connector(t, s)
                                    for t, s in zip(t_channels, s_channels)])
        return connectors

    @staticmethod
    def _build_feature_connector(t_channel, s_channel):
        C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
             nn.BatchNorm2d(t_channel)]
        for m in C:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return nn.Sequential(*C)

    def forward(self, g_s):
        out = []
        for i in range(len(g_s)):
            out.append(self.connectors[i](g_s[i]))

        return out


class ConvReg(nn.Module):
    """Convolutional regression for FitNet"""
    def __init__(self, s_shape, t_shape, use_relu=True):
        super(ConvReg, self).__init__()
        self.use_relu = use_relu
        s_N, s_C, s_H, s_W = s_shape
        t_N, t_C, t_H, t_W = t_shape
        if s_H == 2 * t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1)
        elif s_H >= t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=(1+s_H-t_H, 1+s_W-t_W))
        else:
            raise NotImplemented('student size {}, teacher size {}'.format(s_H, t_H))
        self.bn = nn.BatchNorm2d(t_C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_relu:
            return self.relu(self.bn(x))
        else:
            return self.bn(x)


class Regress(nn.Module):
    """Simple Linear Regression for hints"""
    def __init__(self, dim_in=1024, dim_out=1024):
        super(Regress, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.relu(x)
        return x


class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class LinearEmbed(nn.Module):
    """Linear Embedding"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(LinearEmbed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x


class MLPEmbed(nn.Module):
    """non-linear embed by MLP"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(MLPEmbed, self).__init__()
        self.linear1 = nn.Linear(dim_in, 2 * dim_out)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(2 * dim_out, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.l2norm(self.linear2(x))
        return x


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class Flatten(nn.Module):
    """flatten module"""
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)


class PoolEmbed(nn.Module):
    """pool and embed"""
    def __init__(self, layer=0, dim_out=128, pool_type='avg'):
        super().__init__()
        if layer == 0:
            pool_size = 8
            nChannels = 16
        elif layer == 1:
            pool_size = 8
            nChannels = 16
        elif layer == 2:
            pool_size = 6
            nChannels = 32
        elif layer == 3:
            pool_size = 4
            nChannels = 64
        elif layer == 4:
            pool_size = 1
            nChannels = 64
        else:
            raise NotImplementedError('layer not supported: {}'.format(layer))

        self.embed = nn.Sequential()
        if layer <= 3:
            if pool_type == 'max':
                self.embed.add_module('MaxPool', nn.AdaptiveMaxPool2d((pool_size, pool_size)))
            elif pool_type == 'avg':
                self.embed.add_module('AvgPool', nn.AdaptiveAvgPool2d((pool_size, pool_size)))

        self.embed.add_module('Flatten', Flatten())
        self.embed.add_module('Linear', nn.Linear(nChannels*pool_size*pool_size, dim_out))
        self.embed.add_module('Normalize', Normalize(2))

    def forward(self, x):
        return self.embed(x)

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

