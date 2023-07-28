import logging

import tqdm

import torch
from torch.utils.data import DataLoader
from torch.nn import Module

from ._utils import AverageMeter, accuracy


def validate(val_loader: DataLoader, model: Module, criterion: Module, device: torch.device):
    logger = logging.getLogger("validate")
    logger.info("Start validation")

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for x, target in tqdm.tqdm(val_loader):
            x = x.to(device)
            target = target.to(device)

            # compute output
            output = model(x)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), x.shape[0])
            top1.update(acc1[0], x.shape[0])
            top5.update(acc5[0], x.shape[0])

        logger.info("acc@1: %.4f acc@5: %.4f, loss: %.5f", top1.avg, top5.avg, losses.avg)
    return top1.avg, top5.avg, losses.avg
