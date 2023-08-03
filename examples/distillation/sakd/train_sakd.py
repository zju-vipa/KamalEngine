import os
import argparse
from typing import Dict, Any, List, Iterable
import copy
import logging
import yaml

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import *
from torch.utils.data import DataLoader, Dataset
from torch.nn import Module, ModuleDict
from torch.utils.tensorboard import SummaryWriter

from kamal.vision.datasets.cifar import get_cifar100_dataloaders_sample, get_cifar_10, get_cifar_100
from models import get_model
from distiller_zoo import get_loss_module, get_loss_forward
from models.classifier import LinearClassifier
from models.gs import gumbel_softmax
from models.util import ConvReg, Connector, Paraphraser, Translator, LinearEmbed

from distiller_zoo.FitNet import HintLoss
from distiller_zoo.AT import Attention
from distiller_zoo.crd.criterion import CRDLoss
from distiller_zoo.NST import NSTLoss
from distiller_zoo.SP import Similarity
from distiller_zoo.RKD import RKDLoss
from distiller_zoo.PKT import PKT
from distiller_zoo.KDSVD import KDSVD
from distiller_zoo.CC import Correlation
from distiller_zoo.VID import VIDLoss
from distiller_zoo.AB import ABLoss
from distiller_zoo.FT import FactorTransfer
from distiller_zoo.FSP import FSP

from helper.util import str2bool, get_logger, preserve_memory, adjust_learning_rate_stage
from helper.util import make_deterministic
from helper.util import AverageMeter, accuracy
from helper.validate import validate
from helper.pretrain import init_pretrain
from helper.optim import get_optimizer


def get_dataloader(cfg: Dict[str, Any]):
    # dataset
    dataset_cfg = cfg["dataset"]
    train_dataset = get_dataset(split="train", **dataset_cfg)
    val_dataset = get_dataset(split="val", **dataset_cfg)
    num_classes = len(train_dataset.classes)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg["validation"]["batch_size"],
        num_workers=cfg["validation"]["num_workers"],
        shuffle=False,
        pin_memory=True
    )
    return train_loader, val_loader, num_classes


def get_teacher(cfg: Dict[str, Any], num_classes: int) -> Module:
    teacher_cfg = copy.deepcopy(cfg["kd"]["teacher"])
    teacher_name = teacher_cfg["name"]
    ckpt_fp = teacher_cfg["checkpoint"]
    teacher_cfg.pop("name")
    teacher_cfg.pop("checkpoint")

    # load state dict
    state_dict = torch.load(ckpt_fp, map_location="cpu")["model"]

    model_t = get_model(
        model_name=teacher_name,
        num_classes=num_classes,
        state_dict=state_dict,
        **teacher_cfg
    )
    return model_t


def get_student(cfg: Dict[str, Any], num_classes: int) -> Module:
    student_cfg = copy.deepcopy(cfg["kd"]["student"])
    student_name = student_cfg["name"]
    student_cfg.pop("name")

    state_dict = None
    if "checkpoint" in student_cfg.keys():
        state_dict = torch.load(student_cfg["checkpoint"], map_location="cpu")["model"]
        student_cfg.pop("checkpoint")

    model_s = get_model(
        model_name=student_name,
        num_classes=num_classes,
        state_dict=state_dict,
        **student_cfg
    )
    return model_s


def get_pre_student(cfg: Dict[str, Any], num_classes: int) -> Module:
    student_cfg = copy.deepcopy(cfg["kd"]["prestudent"])
    student_name = student_cfg["name"]
    student_cfg.pop("name")

    state_dict = None
    if "checkpoint" in student_cfg.keys():
        state_dict = torch.load(student_cfg["checkpoint"], map_location="cpu")["model"]
        student_cfg.pop("checkpoint")

    model_s = get_model(
        model_name=student_name,
        num_classes=num_classes,
        state_dict=state_dict,
        **student_cfg
    )
    return model_s


DATASET_DICT = {
    "cifar-10": get_cifar_10,
    "cifar-100": get_cifar_100,
    "cifar10": get_cifar_10,
    "cifar100": get_cifar_100,
    "CIFAR10": get_cifar_10,
    "CIFAR100": get_cifar_100
}


def get_dataset(name: str, root: str, loss_method: str = 'ce', split: str = "train", **kwargs) -> Dataset:
    fn = DATASET_DICT[name]
    return fn(root=root, loss_method=loss_method, split=split)


def train_epoch(
        cfg: Dict[str, Any],
        epoch: int,
        train_loader: DataLoader,
        module_dict: ModuleDict,
        criterion_dict: ModuleDict,
        optimizer: Optimizer,
        agent_optimizer: Optimizer,
        tb_writer: SummaryWriter,
        device: torch.device,
        layers: List
):
    logger = logging.getLogger("train_epoch")
    # setting parameters
    gamma = cfg["kd"]["loss_weights"]["classify_weight"]
    alpha = cfg["kd"]["loss_weights"]["kd_weight"]
    beta = cfg["kd"]["loss_weights"]["other_kd"]

    logger.info(
        "Starting train one epoch with [gamma: %.5f, alpha: %.5f, beta: %.5f]...",
        gamma, alpha, beta
    )

    for name, module in module_dict.items():
        if name == "teacher":
            module.eval()
        else:
            module.train()

    criterion_cls = criterion_dict["cls"]
    criterion_div = criterion_dict["div"]
    criterion_kd = criterion_dict["kd"]

    model_s = module_dict["student"].train()
    model_t = module_dict["teacher"].eval()

    policy = module_dict["policy"].train()

    hints = []
    anti_hints = []
    des_middle_layer = layers

    if cfg["kd_loss"]["name"] in ['CRD', 'FitNet', 'SP', 'CC', 'PKT', "CRD", 'RKD', 'FT']:
        hints.append(module_dict["hint"].train())
        anti_hints.append(module_dict["anti_hint"].train())

        if cfg["kd_loss"]["name"] == "FitNet":
            loss_hint = module_dict["loss_hint"].train()
        elif cfg["kd_loss"]["name"] == 'CC':
            cc_embed_s = module_dict["cc_embed_s"].train()
            cc_embed_t = module_dict["cc_embed_t"].train()
        elif cfg["kd_loss"]["name"] in ['FT']:
            ft_s = module_dict["translator"].train()
            ft_t = module_dict["paraphraser"].train()
    elif cfg["kd_loss"]["name"] in ['AT', 'NST', 'KDSVD', 'VID']:
        for key in module_dict.keys():
            if key.startswith('hint'):
                hints.append(module_dict[key].train())
            if key.startswith('anti_hint'):
                anti_hints.append(module_dict[key].train())
    else:
        raise NotImplementedError(cfg["kd_loss"]["name"])

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    cnt0 = AverageMeter()
    cnt1 = AverageMeter()
    cnt2 = AverageMeter()
    cnt3 = AverageMeter()
    cnt4 = AverageMeter()
    cnts = [cnt0, cnt1, cnt2, cnt3, cnt4]

    for idx, data in enumerate(train_loader):
        __global_values__["it"] += 1

        if cfg["kd_loss"]["name"] in ['CRD']:
            x, target, index, contrast_idx = data
            contrast_idx = contrast_idx.to(device)
            index = index.to(device)
        else:
            x, target = data

        x = x.to(device)
        target = target.to(device)

        # ===================forward=====================

        with torch.no_grad():
            feat_t, logit_t = model_t(x, begin=0, end=100, is_feat=True)

        feat_s, logit_s = model_s(x, begin=0, end=100, is_feat=True)

        middle_layer = [des_middle_layer[i] if des_middle_layer[i] > 0 else min(len(feat_s), len(feat_t)) + des_middle_layer[i]
                        for i in range(len(des_middle_layer))]

        policy_feat = torch.cat((feat_t[-1].clone().detach(), feat_s[-1].clone().detach()), 1)
        policy_res = policy(policy_feat)
        action = gumbel_softmax(policy_res.view(policy_res.size(0), -1, 2), temperature=5)

        loss_cls = torch.tensor(0.0, device=device, dtype=torch.float)
        loss_kd = torch.tensor(0.0, device=device, dtype=torch.float)
        loss_div = torch.tensor(0.0, device=device, dtype=torch.float)

        ac_middle = [action[:, i, 0].contiguous().float().view(-1) for i in range(action.shape[1])]

        if cfg["kd_loss"]["name"] == "FitNet":
            f_s = loss_hint(feat_s[2])
            f_t = feat_t[2]
            choose_index = ac_middle[0].detach() > 0.5
            if logit_s[choose_index].shape[0] > 0:
                loss_kd += criterion_kd(f_s[choose_index], f_t[choose_index].detach()) * logit_s[choose_index].shape[0] / logit_s.shape[0]
        elif cfg["kd_loss"]["name"] == "CRD":
            choose_index = ac_middle[0].detach() > 0.5
            if logit_s[choose_index].shape[0] > 0:
                loss_kd = criterion_kd(feat_s[-1][choose_index], feat_t[-1][choose_index].detach(), index[choose_index],
                                       contrast_idx[choose_index]) * logit_t[choose_index].size(0) / logit_t.size(0)
        elif cfg["kd_loss"]["name"] in ["AT", "NST", "KDSVD"]:
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            middle_choose_list = [each_ac.detach() for each_ac in ac_middle[:-1]]

            g_s = [g_s[feat][middle_choose_list[feat] > 0.5] if
                   g_s[feat][middle_choose_list[feat] > 0.5].shape[0] > 0 else torch.zeros_like(g_s[feat]).cuda()
                   for feat in range(len(g_s))]
            g_t = [g_t[feat][middle_choose_list[feat] > 0.5].detach() if
                   g_t[feat][middle_choose_list[feat] > 0.5].shape[0] > 0 else torch.zeros_like(g_t[feat]).cuda()
                   for feat in range(len(g_t))]
            whole = criterion_kd(g_s, g_t)
            whole = [whole[each_kd] * logit_t[middle_choose_list[each_kd] > 0.5].size(0) / logit_t.size(0) if
                     g_s[each_kd].shape[0] > 0
                     else torch.tensor(0.0, device=device, dtype=torch.float)
                     for each_kd in range(len(whole))]

            loss_kd += sum(whole)
        elif cfg["kd_loss"]["name"] in ["SP"]:
            f_s = feat_s[-2]
            f_t = feat_t[-2]
            choose_index = ac_middle[0].detach() > 0.5
            if logit_s[choose_index].shape[0] > 0:
                loss_kd = sum(criterion_kd([f_s[choose_index]], [f_t[choose_index].detach()])) * logit_s[choose_index].shape[0] / \
                    logit_s.shape[0]
        elif cfg["kd_loss"]["name"] in ["FT"]:
            factor_s = ft_s(feat_s[-2])
            factor_t = ft_t(feat_t[-2], is_factor=True)
            loss_kd = criterion_kd(factor_s, factor_t)
            choose_index = ac_middle[0].detach() > 0.5
            if logit_s[choose_index].shape[0] > 0:
                loss_kd = criterion_kd(factor_s[choose_index], factor_t[choose_index].detach()) * logit_s[choose_index].shape[
                    0] / logit_s.shape[0]
        elif cfg["kd_loss"]["name"] in ["RKD", 'PKT']:
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            choose_index = ac_middle[0].detach() > 0.5
            if logit_s[choose_index].shape[0] > 0:
                loss_kd = criterion_kd(f_s[choose_index], f_t[choose_index].detach()) * logit_s[choose_index].shape[
                    0] / logit_s.shape[0]
        elif cfg["kd_loss"]["name"] in ['CC']:
            f_s = cc_embed_s(feat_s[-1])
            f_t = cc_embed_t(feat_t[-1])
            choose_index = ac_middle[0].detach() > 0.5
            if logit_s[choose_index].shape[0] > 0:
                loss_kd = criterion_kd(f_s[choose_index], f_t[choose_index].detach()) * logit_s[choose_index].shape[
                    0] / logit_s.shape[0]
        elif cfg["kd_loss"]["name"] in ['VID']:
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            middle_choose_list = [each_ac.detach() for each_ac in ac_middle[:-1]]

            loss_group = [criterion_kd[each_f](g_s[each_f][middle_choose_list[each_f] > 0.5],
                                               g_t[each_f][middle_choose_list[each_f] > 0.5].detach()) *
                          logit_t[middle_choose_list[each_f] > 0.5].size(0) / logit_t.size(0)
                          if g_t[each_f][middle_choose_list[each_f] > 0.5].shape[0] > 0
                          else torch.tensor(0.0, device=device, dtype=torch.float)
                          for each_f in range(len(g_s))]
            loss_kd = sum(loss_group)
        else:
            raise NotImplementedError(cfg["kd_loss"]["name"])

        # ========================== Graft ===========================================
        feat_to_s = x.clone()
        feat_to_t = x.clone()
        model_s.eval()

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

        # ============================ Output ============================================
        ac = ac_middle[-1]
        out_ac = ac.view(-1, 1)
        logit = feat_middle_s * (1 - out_ac) + feat_middle_t * out_ac  # graft output

        # ======================== CE + DIV =========================================
        model_s.train()
        loss_cls = criterion_cls(logit, target)

        last_choose = ac.detach()

        choose_index = last_choose > 0.5
        if logit_s[choose_index].shape[0] > 0:
            loss_div += (criterion_div(logit_s[choose_index], logit_t[choose_index].detach()) * logit_s[choose_index].shape[0] / logit_s.shape[0])

        loss_ori_cls = criterion_cls(logit_s, target)

        if loss_cls > loss_ori_cls:
            loss_cls *= gamma

        if loss_cls < loss_ori_cls:
            loss_cls /= gamma

        loss_cls += max(0, logit_s.shape[0] / 2 - sum(ac_middle[-1]))

        loss_div = alpha * loss_div
        loss = loss_ori_cls + (beta * loss_kd + loss_div)

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), x.shape[0])
        top1.update(acc1[0], x.shape[0])
        top5.update(acc5[0], x.shape[0])

        # ===================backward=====================
        agent_optimizer.zero_grad()
        loss_cls.backward(retain_graph=True)
        agent_optimizer.step()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print info
        tb_writer.add_scalars(
            main_tag="train/acc",
            tag_scalar_dict={
                "@1": acc1,
                "@5": acc5,
            },
            global_step=__global_values__["it"]
        )
        tb_writer.add_scalars(
            main_tag="train/loss",
            tag_scalar_dict={
                "cls": loss_ori_cls.item(),
                "div": loss_div.item(),
                "kd": loss_kd.item()
            },
            global_step=__global_values__["it"]
        )
        writer_dict = dict()
        for graft_layer in range(len(middle_layer) - 1):
            writer_dict['graft_cnt{}'.format(graft_layer)] = logit_t[ac_middle[graft_layer] > 0.5].shape[0] / logit_t.shape[0]
            cnts[graft_layer].update(logit_t[ac_middle[graft_layer] > 0.5].shape[0] / logit_t.shape[0]*100, logit_t.shape[0])

        tb_writer.add_scalars(
            main_tag="train_gate_num",
            tag_scalar_dict=writer_dict,
            global_step=__global_values__["it"]
        )
        if idx % cfg["training"]["print_iter_freq"] == 0:
            tup = (loss_cls.item(), loss_ori_cls.item(), loss_div.item(), loss_kd.item())
            msg = "graftloss_cls: %.5f, loss_cls: %.5f, loss_div: %.5f, loss_kd: %.5f"

            for cnt_ in range(len(ac_middle)):
                msg += ", cnt"+str(cnt_)+": %d"
                tup = tup + (logit_t[ac_middle[cnt_] > 0.5].shape[0],)

            logger.info(msg, *tup)

    return top1.avg, losses.avg


def train_kd(
        cfg: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        module_dict: ModuleDict,
        criterion_dict: ModuleDict,
        optimizer: Optimizer,
        lr_scheduler: MultiStepLR,
        agent_optimizer: Optimizer,
        agent_lr_scheduler: MultiStepLR,
        tb_writer: SummaryWriter,
        device: torch.device,
        ckpt_dir: str,
        layers: List
):
    logger = logging.getLogger("train")
    logger.info("Start training...")

    best_acc = 0
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        adjust_learning_rate_stage(
            optimizer=optimizer,
            cfg=cfg,
            epoch=epoch
        )
        print(cfg["kd"]["teacher"]["name"], cfg["kd"]["student"]["name"])
        logger.info("Start training epoch: %d, current lr: %.6f",
                    epoch, lr_scheduler.get_last_lr()[0])
        logger.info("Start training epoch: %d, current agent lr: %.6f",
                    epoch, agent_lr_scheduler.get_last_lr()[0])

        train_acc, train_loss = train_epoch(
            cfg=cfg,
            epoch=epoch,
            train_loader=train_loader,
            module_dict=module_dict,
            criterion_dict=criterion_dict,
            optimizer=optimizer,
            agent_optimizer=agent_optimizer,
            tb_writer=tb_writer,
            device=device,
            layers=layers
        )

        tb_writer.add_scalar("epoch/train_acc", train_acc, epoch)
        tb_writer.add_scalar("epoch/train_loss", train_loss, epoch)

        val_acc, val_acc_top5, val_loss = validate(
            val_loader=val_loader,
            model=module_dict["student"],
            criterion=criterion_dict["cls"],
            device=device
        )

        tb_writer.add_scalar("epoch/val_acc", val_acc, epoch)
        tb_writer.add_scalar("epoch/val_loss", val_loss, epoch)
        tb_writer.add_scalar("epoch/val_acc_top5", val_acc_top5, epoch)

        logger.info(
            "Epoch: %04d | %04d, acc: %.4f, loss: %.5f, val_acc: %.4f, val_acc_top5: %.4f, val_loss: %.5f",
            epoch, cfg["training"]["epochs"],
            train_acc, train_loss,
            val_acc, val_acc_top5, val_loss,
        )

        lr_scheduler.step()
        agent_lr_scheduler.step()

        state = {
            "epoch": epoch,
            "model": module_dict["student"].state_dict(),
            "policy": module_dict["policy"].state_dict(),
            "acc": val_acc,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict()
        }

        if cfg["kd_loss"]["name"] == "FitNet":
            state["loss_hint"] = module_dict["loss_hint"].state_dict()
        elif cfg["kd_loss"]["name"] == "CRD":
            state["embed_s"] = module_dict["crd_embed_s"].state_dict()
            state["embed_t"] = module_dict["crd_embed_t"].state_dict()
        elif cfg["kd_loss"]["name"] == "CC":
            state["embed_s"] = module_dict["cc_embed_s"].state_dict()
            state["embed_t"] = module_dict["cc_embed_t"].state_dict()

        # # regular saving
        # if epoch % cfg["training"]["save_ep_freq"] == 0:
        #     logger.info("Saving epoch %d checkpoint...", epoch)
        #     save_file = os.path.join(ckpt_dir, "epoch_{}.pth".format(epoch))
        #     torch.save(state, save_file)

        # save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_ep = epoch

            save_file = os.path.join(ckpt_dir, "best.pth")
            logger.info("Saving the best model with acc: %.4f", best_acc)
            torch.save(state, save_file)

        logger.info("Epoch: %04d | %04d, best acc: %.4f,", epoch, cfg["training"]["epochs"], best_acc)

    logger.info("Final best accuracy: %.5f, at epoch: %d", best_acc, best_ep)


def main(
        cfg_filepath: str,
        file_name_cfg: str,
        logdir: str,
        gpu_preserve: bool = False,
        debug: bool = False
):
    with open(cfg_filepath) as fp:
        cfg = yaml.load(fp, Loader=yaml.SafeLoader)

    seed = cfg["training"]["seed"]

    ckpt_dir = os.path.join(logdir, "ckpt")
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    formatter = (
        cfg["kd"]["teacher"]["name"],
        cfg["kd"]["student"]["name"],
        cfg["kd_loss"]["T"],
        cfg["dataset"]["name"],
    )
    writer = SummaryWriter(
        log_dir=os.path.join(
            logdir,
            "tf-logs",
            file_name_cfg.format(*formatter)
        ),
        flush_secs=1
    )

    train_log_dir = os.path.join(logdir, "train-logs")
    os.makedirs(train_log_dir, exist_ok=True)
    logger = get_logger(
        level=logging.INFO,
        mode="w",
        name=None,
        logger_fp=os.path.join(
            train_log_dir,
            "training-" + file_name_cfg.format(*formatter) + ".log"
        )
    )
    logger.info("Start running with config: \n{}".format(yaml.dump(cfg)))

    # set seed
    make_deterministic(seed)
    logger.info("Set seed : {}".format(seed))

    if gpu_preserve:
        logger.info("Preserving memory...")
        preserve_memory(args.preserve_percent)
        logger.info("Preserving memory done")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get dataloaders
    logger.info("Loading datasets...")
    if cfg["kd_loss"]["name"] in ['CRD'] and cfg["dataset"]["name"] in ["cifar100", "CIFAR100", "cifar-100"]:
        # prepare data
        train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(
            batch_size=cfg["training"]["batch_size"],
            num_workers=cfg["training"]["num_workers"],
            k=16384,
            mode='exact'
        )
        num_classes = 100
    else:
        train_loader, val_loader, num_classes = get_dataloader(cfg)
    logger.info("num_classes: {}".format(num_classes))

    # get models
    logger.info("Loading teacher and student...")
    model_t = get_teacher(cfg, num_classes).to(device) # teacher model
    model_s = get_student(cfg, num_classes).to(device) # student model

    model_t.eval()
    model_s.eval()

    if cfg["dataset"]["name"] in ["tiny-imagenet"]:
        data = torch.randn(2, 3, 64, 64).to(device)
    else:
        data = torch.randn(2, 3, 32, 32).to(device)

    feat_t, _ = model_t(data, begin=0, end=100, is_feat=True)
    feat_s, _ = model_s(data, begin=0, end=100, is_feat=True)

    for p in model_t.parameters():
        p.requires_grad = False

    logger.info(model_s)

    module_dict = nn.ModuleDict(dict(
        teacher=model_t,
        student=model_s
    ))
    trainable_dict = nn.ModuleDict(dict(
        student=model_s,
    ))

    trainable_dict2 = nn.ModuleDict(dict(
    ))

    # get loss modules
    criterion_dict, loss_trainable_dict = get_loss_module(
        cfg=cfg,
        module_dict=module_dict,
        train_loader=train_loader,
        tb_writer=writer,
        device=device
    )
    trainable_dict.update(loss_trainable_dict)

    # ======================= other_kd ==========================================
    # graft + policy
    layers = [0, 100]
    if cfg["kd_loss"]["name"] in ['AT', 'NST', 'VID', 'KDSVD', 'AB']:
        feat_t_lis = feat_t[1:-1]
        feat_s_lis = feat_s[1:-1]
        real_len = min(len(feat_s_lis), len(feat_t_lis))
        layers = [0] + [i + 1 for i in range(real_len)] + [100]
        policy = LinearClassifier(model_t.get_hint_channel() + model_s.get_hint_channel(), real_len * 2 + 2).to(device)

        for each_feat in range(real_len):
            print(feat_s_lis[each_feat].shape, feat_t_lis[each_feat].shape)
            module_dict["hint_{}".format(each_feat)] = ConvReg(feat_s_lis[each_feat].shape, feat_t_lis[each_feat].shape).to(device)
            module_dict["anti_hint_{}".format(each_feat)] = ConvReg(feat_t_lis[each_feat].shape, feat_s_lis[each_feat].shape).to(device)
            trainable_dict2["hint_{}".format(each_feat)] = module_dict["hint_{}".format(each_feat)]
            trainable_dict2["anti_hint_{}".format(each_feat)] = module_dict["anti_hint_{}".format(each_feat)]
    elif cfg["kd_loss"]["name"] in ['FSP']:
        feat_t_lis = feat_t[:-1]
        feat_s_lis = feat_s[:-1]
        real_len = min(len(feat_s_lis), len(feat_t_lis))
        layers = [0, 0] + [i + 1 for i in range(real_len)] + [100]
        policy = LinearClassifier(model_t.get_hint_channel() + model_s.get_hint_channel(), real_len * 2 + 2).to(device)

        for each_feat in range(real_len):
            module_dict["hint_{}".format(each_feat)] = ConvReg(feat_s_lis[each_feat].shape, feat_t_lis[each_feat].shape).to(device)
            module_dict["anti_hint_{}".format(each_feat)] = ConvReg(feat_t_lis[each_feat].shape, feat_s_lis[each_feat].shape).to(device)
            trainable_dict2["hint_{}".format(each_feat)] = module_dict["hint_{}".format(each_feat)]
            trainable_dict2["anti_hint_{}".format(each_feat)] = module_dict["anti_hint_{}".format(each_feat)]

    elif cfg["kd_loss"]["name"] in ['FitNet']:  # 2
        policy = LinearClassifier(model_t.get_hint_channel() + model_s.get_hint_channel(), 4).to(device)
        layers = [0, 2, 100]
        regress_s = ConvReg(feat_s[2].shape, feat_t[2].shape).to(device)
        anti_regress_s = ConvReg(feat_t[2].shape, feat_s[2].shape).to(device)
        module_dict["hint"] = regress_s
        module_dict["anti_hint"] = anti_regress_s
        trainable_dict2["hint"] = regress_s
        trainable_dict2["anti_hint"] = anti_regress_s
    elif cfg["kd_loss"]["name"] in ['CRD', 'RKD', 'PKT', 'CC']:  # -1
        policy = LinearClassifier(model_t.get_hint_channel() + model_s.get_hint_channel(), 4).to(device)
        layers = [0, -1, 100]
        regress_s = LinearClassifier(feat_s[-1].shape[1], feat_t[-1].shape[1]).to(device)
        anti_regress_s = LinearClassifier(feat_t[-1].shape[1], feat_s[-1].shape[1]).to(device)
        module_dict["hint"] = regress_s
        module_dict["anti_hint"] = anti_regress_s
        trainable_dict2["hint"] = regress_s
        trainable_dict2["anti_hint"] = anti_regress_s
    elif cfg["kd_loss"]["name"] in ['FT', 'SP']:  # -2
        policy = LinearClassifier(model_t.get_hint_channel() + model_s.get_hint_channel(), 4).to(device)
        layers = [0, -2, 100]
        regress_s = ConvReg(feat_s[-2].shape, feat_t[-2].shape).to(device)
        anti_regress_s = ConvReg(feat_t[-2].shape, feat_s[-2].shape).to(device)
        module_dict["hint"] = regress_s
        module_dict["anti_hint"] = anti_regress_s
        trainable_dict2["hint"] = regress_s
        trainable_dict2["anti_hint"] = anti_regress_s
    else:
        raise NotImplementedError(cfg["kd_loss"]["name"])

    if cfg["kd_loss"]["homo"]:
        policy.set_n_to_True()

    policy.eval()
    module_dict["policy"] = policy
    trainable_dict2["policy"] = policy

    if cfg["kd_loss"]["name"] == "FitNet":
        criterion_dict["kd"] = HintLoss().to(device)
        loss_hint = ConvReg(feat_s[2].shape, feat_t[2].shape).to(device)
        module_dict["loss_hint"] = loss_hint
        trainable_dict["loss_hint"] = loss_hint
    elif cfg["kd_loss"]["name"] == "CRD":
        criterion_dict["kd"] = CRDLoss(
            s_dim=feat_s[-1].shape[1],
            t_dim=feat_t[-1].shape[1],
            feat_dim=cfg["kd_loss"]["feat_dim"],
            n_data=n_data
        ).to(device)
        trainable_dict["crd_embed_s"] = criterion_dict["kd"].embed_s
        trainable_dict["crd_embed_t"] = criterion_dict["kd"].embed_t
        module_dict["crd_embed_s"] = criterion_dict["kd"].embed_s
        module_dict["crd_embed_t"] = criterion_dict["kd"].embed_t
    elif cfg["kd_loss"]["name"] == "AT":
        criterion_dict["kd"] = Attention().to(device)
    elif cfg["kd_loss"]["name"] == "NST":
        criterion_dict["kd"] = NSTLoss().to(device)
    elif cfg["kd_loss"]["name"] == "SP":
        criterion_dict["kd"] = Similarity().to(device)
    elif cfg["kd_loss"]["name"] == "RKD":
        criterion_dict["kd"] = RKDLoss().to(device)
    elif cfg["kd_loss"]["name"] == "PKT":
        criterion_dict["kd"] = PKT().to(device)
    elif cfg["kd_loss"]["name"] == "KDSVD":
        criterion_dict["kd"] = KDSVD().to(device)
    elif cfg["kd_loss"]["name"] == "CC":
        criterion_dict["kd"] = Correlation().to(device)
        embed_s = LinearEmbed(feat_s[-1].shape[1], cfg["kd_loss"]["feat_dim"]).to(device)
        embed_t = LinearEmbed(feat_t[-1].shape[1], cfg["kd_loss"]["feat_dim"]).to(device)
        module_dict["cc_embed_s"] = embed_s
        module_dict["cc_embed_t"] = embed_t
        trainable_dict["cc_embed_s"] = embed_s
        trainable_dict["cc_embed_t"] = embed_t
    elif cfg["kd_loss"]["name"] == 'VID':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = nn.ModuleList(
            [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
        ).to(device)
        criterion_dict["kd"] = criterion_kd
        trainable_dict["vid"] = criterion_kd
    elif cfg["kd_loss"]["name"] == "FT":
        s_shape = feat_s[-2].shape
        t_shape = feat_t[-2].shape
        paraphraser = Paraphraser(t_shape).to(device)
        translator = Translator(s_shape, t_shape).to(device)

        # init stage training
        init_trainable_dict = nn.ModuleDict(dict(
            paraphraser=paraphraser,
        ))
        criterion_init = nn.MSELoss().to(device)

        init_pretrain(cfg, module_dict, init_trainable_dict, criterion_init, train_loader, logger, device)

        # classification
        criterion_dict["kd"] = FactorTransfer().to(device)

        trainable_dict["translator"] = translator
        module_dict["translator"] = translator
        module_dict["paraphraser"] = paraphraser
    else:
        raise NotImplementedError(cfg["kd_loss"]["name"])

    assert "teacher" not in trainable_dict.keys(), "teacher is not trainable"

    # optimizer
    optimizer = torch.optim.SGD(
        params=trainable_dict.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["optimizer"]["weight_decay_stage2"],
        momentum=cfg["training"]["optimizer"]["momentum"])

    lr_scheduler = MultiStepLR(
        optimizer=optimizer,
        milestones=cfg["training"]["lr_decay_epochs"],
        gamma=cfg["training"]["lr_decay_rate"]
    )
    agent_optimizer = torch.optim.SGD(
        params=trainable_dict2.parameters(),
        lr=cfg["training"]["lr"]/2,
        weight_decay=cfg["training"]["optimizer"]["weight_decay_agent"],
        momentum=cfg["training"]["optimizer"]["momentum"])

    agent_lr_scheduler = MultiStepLR(
        optimizer=agent_optimizer,
        milestones=cfg["training"]["lr_decay_epochs"],
        gamma=cfg["training"]["lr_decay_rate"]
    )

    module_dict["teacher"] = model_t.to(device)
    logger.info(optimizer)
    logger.info(agent_lr_scheduler)

    train_kd(
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        module_dict=module_dict,
        criterion_dict=criterion_dict,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        agent_optimizer=agent_optimizer,
        agent_lr_scheduler=agent_lr_scheduler,
        tb_writer=writer,
        device=device,
        ckpt_dir=ckpt_dir,
        layers=layers
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--logdir", type=str)
    parser.add_argument("--file_name_cfg", type=str)
    parser.add_argument("--gpu_preserve", type=str2bool, default=False)
    parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument("--preserve_percent", type=float, default=0.95)
    args = parser.parse_args()

    __global_values__ = dict(it=0)
    main(
        cfg_filepath=args.config,
        file_name_cfg=args.file_name_cfg,
        logdir=args.logdir,
        gpu_preserve=args.gpu_preserve,
        debug=args.debug
    )
