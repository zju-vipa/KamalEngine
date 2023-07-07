import logging

import tqdm
from typing import Dict, Any, List

import torch
from torch.utils.data import DataLoader
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, ModuleDict
from models.gs import gumbel_softmax
from .util import AverageMeter, accuracy

def validate_OKDD(
        val_loader: DataLoader,
        model: Module,
        criterion_cls: Module,
        criterion_div: Module,
        device: torch.device,
        consistency_weight: float,
        num_branches: int
):
    logger = logging.getLogger("validate")
    logger.info("Start validation")

    losses = AverageMeter()
    dist_avg = AverageMeter()
    top1 = list(range(num_branches + 1))
    top5 = list(range(num_branches + 1))
    for i in range(num_branches + 1):
        top1[i] = AverageMeter()
        top5[i] = AverageMeter()

    # switch to evaluate mode
    model.eval()
    pdist = nn.PairwiseDistance(p=2)

    with torch.no_grad():
        for x, target in tqdm.tqdm(val_loader):
            x = x.to(device)
            target = target.to(device)

            output_batch, x_m, x_stu = model(x)

            loss_true = 0
            loss_group = 0
            for i in range(num_branches - 1):
                loss_true += criterion_cls(output_batch[:, :, i], target)
                loss_group += criterion_div(output_batch[:, :, i], x_m[:, :, i])
            # loss_true = loss_true / args.num_branches
            # loss_group = loss_group / args.num_branches
            loss = loss_true + criterion_cls(x_stu, target) + consistency_weight * (
                    loss_group + criterion_div(x_stu, torch.mean(output_batch, dim=2)))

            losses.update(loss.item(), x.shape[0])

            for i in range(num_branches - 1):
                metrics = accuracy(output_batch[:, :, i], target, topk=(1, 5))
                top1[i].update(metrics[0].item())
                top5[i].update(metrics[1].item())

            metrics = accuracy(x_stu, target, topk=(1, 5))
            top1[num_branches - 1].update(metrics[0].item())
            top5[num_branches - 1].update(metrics[1].item())

            e_metrics = accuracy(torch.mean(output_batch, dim=2), target, topk=(1, 5))
            top1[num_branches].update(e_metrics[0].item())
            top5[num_branches].update(e_metrics[1].item())

            len_kk = output_batch.size(0)
            output_batch = F.softmax(output_batch, dim=1)
            for kk in range(len_kk):
                ret = output_batch[kk, :, :]
                # ret = ret.squeeze(0)
                ret = ret.t()  # branches x classes
                sim = 0
                for j in range(num_branches - 1):
                    for k in range(j + 1, num_branches - 1):
                        sim += pdist(ret[j:j + 1, :], ret[k:k + 1, :])
                sim = sim / 3
                dist_avg.update(sim.item())

        mean_test_accTop1 = 0
        mean_test_accTop5 = 0
        for i in range(num_branches - 1):
            mean_test_accTop1 += top1[i].avg
            mean_test_accTop5 += top5[i].avg
        mean_test_accTop1 /= (num_branches - 1)
        mean_test_accTop5 /= (num_branches - 1)
        # compute mean of all metrics in summary

        logger.info("acc@1: %.4f acc@5: %.4f, loss: %.5f", mean_test_accTop1, mean_test_accTop5, losses.avg)
    return mean_test_accTop1, mean_test_accTop5, losses.avg

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
            output= model(x)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), x.shape[0])
            top1.update(acc1[0], x.shape[0])
            top5.update(acc5[0], x.shape[0])

        logger.info("acc@1: %.4f acc@5: %.4f, loss: %.5f", top1.avg, top5.avg, losses.avg)
    return top1.avg, top5.avg, losses.avg

def validate_policy(cfg: Dict[str, Any], val_loader: DataLoader, module_dict: ModuleDict, criterion: Module, device: torch.device, layers: List):
    logger = logging.getLogger("validate")
    logger.info("Start validation")

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model_s = module_dict["student"].eval()
    model_t = module_dict["teacher"].eval()
    policy = module_dict["policy"].eval()

    hints = []
    anti_hints = []

    if cfg["kd_loss"]["name"] in ['CRD', 'FitNet', 'SP', 'CC', 'PKT', "CRD", 'RKD', 'FT']:
        hints.append(module_dict["hint"].eval())
        anti_hints.append(module_dict["anti_hint"].eval())

    elif cfg["kd_loss"]["name"] in ['AT', 'NST', 'KDSVD', 'VID']:
        for key in module_dict.keys():
            if key.startswith('hint'):
                hints.append(module_dict[key].eval())
            if key.startswith('anti_hint'):
                anti_hints.append(module_dict[key].eval())
    else:
        raise NotImplementedError(cfg["kd_loss"]["name"])

    with torch.no_grad():
        for x, target in tqdm.tqdm(val_loader):
            x = x.to(device)
            target = target.to(device)

            with torch.no_grad():
                feat_t, logit_t = model_t(x, begin=0, end=100, is_feat=True)

                feat_s, logit_s = model_s(x, begin=0, end=100, is_feat=True)

            policy_feat = torch.cat((feat_t[-1].clone().detach(), feat_s[-1].clone().detach()), 1)
            policy_res = policy(policy_feat)
            action = gumbel_softmax(policy_res.view(policy_res.size(0), -1, 2), temperature=5)
            middle_layer = [
                layers[i] if layers[i] > 0 else min(len(feat_s), len(feat_t)) + layers[i]
                for i in range(len(layers))]
            ac_middle = [action[:, i, 0].contiguous().float().view(-1) for i in range(action.shape[1])]
            feat_to_s = x.clone()
            feat_to_t = x.clone()
            for graft_layer in range(len(middle_layer) - 1):

                if graft_layer == 0:
                    be = 0
                else:
                    be = middle_layer[graft_layer] + 1

                feat_middle_s = model_s(feat_to_s, begin=be, end=middle_layer[graft_layer + 1])
                feat_middle_t = model_t(feat_to_t, begin=be, end=middle_layer[graft_layer + 1])

                if graft_layer != len(middle_layer) - 2:
                    if len(feat_middle_s.size()) == 4:
                        feat_ac = ac_middle[graft_layer].view(-1, 1, 1, 1)
                    else:
                        feat_ac = ac_middle[graft_layer].view(-1, 1)

                    feat_to_t = (1 - feat_ac) * hints[graft_layer](feat_middle_s) + feat_ac * feat_middle_t
                    feat_to_s = (1 - feat_ac) * feat_middle_s + feat_ac * anti_hints[graft_layer](feat_middle_t)

            ac = ac_middle[-1]
            out_ac = ac.view(-1, 1)
            logit = feat_middle_s * (1 - out_ac) + feat_middle_t * out_ac  # graft output

            loss = criterion(logit, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(logit, target, topk=(1, 5))
            losses.update(loss.item(), x.shape[0])
            top1.update(acc1[0], x.shape[0])
            top5.update(acc5[0], x.shape[0])

        logger.info("Graft acc@1: %.4f Graft acc@5: %.4f, Graft loss: %.5f", top1.avg, top5.avg, losses.avg)
    return top1.avg, top5.avg, losses.avg
