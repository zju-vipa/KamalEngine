import os
import argparse
from typing import Dict, Any
import copy
import logging

import yaml

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter

from kamal import get_logger, make_deterministic, preserve_memory, str2bool, get_logger_
from kamal.core.tasks.loss import get_loss_module
from kamal.optim import get_optimizer
from kamal.slim.distillation.TDD.tdd import TDDistiller
from kamal.vision.datasets import get_dataset
from kamal.vision.models.classification import get_model

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


def main(
    cfg_filepath: str,
    file_name_cfg: str,
    logdir: str,
    gpu_preserve: bool = False,
    debug: bool = False
):
    with open(cfg_filepath) as fp:
        cfg = yaml.load(fp, Loader=yaml.SafeLoader)

    if debug:
        cfg["training"]["num_workers"] = 0
        cfg["validation"]["num_workers"] = 0

    seed = cfg["training"]["seed"]

    ckpt_dir = os.path.join(logdir, "ckpt")
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    formatter = (
        cfg["kd"]["teacher"]["name"],
        cfg["kd"]["student"]["name"],
        cfg["kd_loss"]["name"],
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
    logger = get_logger_(
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
    train_loader, val_loader, num_classes = get_dataloader(cfg)

    # get models
    logger.info("Loading teacher and student...")
    model_t = get_teacher(cfg, num_classes).to(device)
    model_s = get_student(cfg, num_classes).to(device)
    model_t.eval()
    model_s.eval()

    module_dict = nn.ModuleDict(dict(
        student=model_s,
        teacher=model_t
    ))
    trainable_dict = nn.ModuleDict(dict(student=model_s))

    # get loss modules
    criterion_dict, loss_trainable_dict = get_loss_module(
        cfg=cfg,
        module_dict=module_dict,
        train_loader=train_loader,
        tb_writer=writer,
        device=device
    )
    trainable_dict.update(loss_trainable_dict)

    assert "teacher" not in trainable_dict.keys(), "teacher is not trainable"
    # optimizer
    optimizer = get_optimizer(trainable_dict.parameters(), cfg["training"]["optimizer"])
    lr_scheduler = MultiStepLR(
        optimizer=optimizer,
        milestones=cfg["training"]["lr_decay_epochs"],
        gamma=cfg["training"]["lr_decay_rate"]
    )

    # append teacher after optimizer to avoid weight_decay
    module_dict["teacher"] = model_t.to(device)

    tddistiller = TDDistiller(logger, writer)

    tddistiller.setup(cfg=cfg, train_loader=train_loader, val_loader=val_loader, module_dict=module_dict, \
                    criterion_dict=criterion_dict, optimizer=optimizer, lr_scheduler=lr_scheduler, device=device, ckpt_dir=ckpt_dir)

    tddistiller.run(start_iter=0, max_iter=len(train_loader) * (cfg["training"]["epochs"] + 1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--logdir", type=str)
    parser.add_argument("--file_name_cfg", type=str)
    parser.add_argument("--gpu_preserve", type=str2bool, default=False)
    parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument("--preserve_percent", type=float, default=0.95)
    args = parser.parse_args()

    main(
        cfg_filepath=args.config,
        file_name_cfg=args.file_name_cfg,
        logdir=args.logdir,
        gpu_preserve=args.gpu_preserve,
        debug=args.debug
    )
