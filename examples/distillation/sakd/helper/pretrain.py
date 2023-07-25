import logging
from typing import Dict, Any, List

import tqdm

import torch
from torch import Tensor
from torch.nn import Module, ModuleDict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .util import AverageMeter
from .optim import get_optimizer


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
    AB=abound_init_loss_forward,
    FT=factor_init_loss_forward,
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
        # tb_writer.add_scalars(
        #     main_tag="epoch/init/loss",
        #     tag_scalar_dict={
        #         "init_loss": losses.avg
        #     },
        #     global_step=epoch
        # )
        logger.info("Epoch: [%3d|%3d], loss: %.5f", epoch, cfg["pretrain"]["epochs"], losses.val)
