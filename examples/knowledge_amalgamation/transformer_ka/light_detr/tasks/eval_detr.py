import os
import logging
import time
from logging.handlers import QueueHandler
from typing import Dict, Any
import datetime
import yaml

import torch
import torch.cuda
from torch import nn
import torch.distributed as dist
import torch.utils.data as data
import torch.backends.cudnn

import cv_lib.utils as utils
import cv_lib.distributed.utils as dist_utils

from light_detr.models import build_detr
from light_detr.loss import build_set_criterion, SetCriterion
import light_detr.utils as detr_utils
from light_detr.eval import EvaluationBase, get_evaluator
from light_detr.data import build_eval_dataset


class Evaluator:
    def __init__(
        self,
        val_loader: data.DataLoader,
        model: nn.Module,
        loss: SetCriterion,
        evaluator: EvaluationBase,
        distributed: bool,
        device: torch.device
    ):
        # set up logger
        self.logger = logging.getLogger("trainer_rank_{}".format(dist_utils.get_rank()))

        self.val_loader = val_loader
        self.model = model
        self.loss = loss
        self.evaluator = evaluator
        self.distributed = distributed
        self.device = device

    def validate_and_save(self):
        self.logger.info("Start evaluation")
        ret_dict = self.evaluator(self.model)

        if dist_utils.is_main_process():
            self.logger.info("evaluation done")
            loss = ret_dict["loss"]
            loss_dict = ret_dict["loss_dict"]
            performance: Dict[str, torch.Tensor] = ret_dict["performance"]
            # write logger
            info = "Validation loss: {:.5f}, mean AP: {:.4f}, mean AP50: {:.4f}, mean AP75: {:.4f}\nloss dict: {}\nAP: {}"
            info = info.format(
                loss,
                performance["mean_ap"].item(),
                performance["mean_ap_50"].item(),
                performance["mean_ap_75"].item(),
                utils.to_json_str(utils.tensor_dict_items(loss_dict)),
                utils.to_json_str(utils.tensor_to_list(performance["ap"])),
            )
            self.logger.info(info)
        dist_utils.barrier()

    def __call__(self):
        start_time = time.time()
        # start one epoch
        self.validate_and_save()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info("Validation time %s", total_time_str)


def validate_worker(
    gpu_id: int,
    launch_args: detr_utils.DistLaunchArgs,
    log_args: detr_utils.LogArgs,
    global_cfg: Dict[str, Any],
    resume: str = ""
):
    """
    What created in this function is only used in this process and not shareable
    """
    if launch_args.distributed:
        root_logger = logging.getLogger()
        handler = QueueHandler(log_args.logger_queue)
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)
        root_logger.propagate = False
    # split configs
    data_cfg: Dict[str, Any] = global_cfg["dataset"]
    train_cfg: Dict[str, Any] = global_cfg["training"]
    val_cfg: Dict[str, Any] = global_cfg["validation"]
    if "student" in global_cfg.keys():
        global_cfg["model"] = global_cfg.pop("student")
    model_cfg: Dict[str, Any] = global_cfg["model"]
    loss_cfg: Dict[str, Any] = global_cfg["loss"]
    # set debug number of workers
    if launch_args.debug:
        train_cfg["num_workers"] = 0
        val_cfg["num_workers"] = 0
    distributed = launch_args.distributed
    # get current rank
    current_rank = launch_args.rank
    if distributed:
        if launch_args.multiprocessing:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes == dist_utils.get_rank()
            current_rank = launch_args.rank * launch_args.ngpus_per_node + gpu_id
        dist.init_process_group(
            backend=launch_args.backend,
            init_method=launch_args.master_url,
            world_size=launch_args.world_size,
            rank=current_rank
        )

    assert dist_utils.get_rank() == current_rank, "code bug"
    # set up process logger
    logger = logging.getLogger("worker_rank_{}".format(current_rank))

    # set device
    logger.info("Use GPU: %d for evaluation", gpu_id)
    device = torch.device("cuda:{}".format(gpu_id))
    # IMPORTANT! for distributed validation (reduce_all_object)
    torch.cuda.set_device(device)

    if current_rank == 0:
        logger.info("Starting with configs:\n%s", yaml.dump(global_cfg))

    # make determinstic
    if launch_args.seed is not None:
        seed = launch_args.seed + current_rank
        logger.info("Initial rank %d with seed: %d", current_rank, seed)
        utils.make_deterministic(seed)
    torch.backends.cudnn.benchmark = True

    # get dataloader
    logger.info("Building dataset...")
    val_loader, n_classes = build_eval_dataset(
        data_cfg=data_cfg,
        val_cfg=val_cfg,
        launch_args=launch_args,
    )
    # create model
    logger.info("Building detr...")
    model = build_detr(model_cfg, n_classes, seg=False)
    logger.info("Loading checkpoint...")
    if os.path.isfile(resume):
        resume = torch.load(resume, map_location="cpu")
        if "student" in resume:
            resume["model"] = resume.pop("student")
        if "model" in resume and current_rank == 0:
            try:
                ap = utils.tensor_to_list(resume["performance"]["ap"], 4)
                logger.info(
                    "Loaded checkpoint from epoch: %d, iter: %d, mean AP: %.4f, mean AP50: %.4f, AP:\n%s",
                    resume["epoch"],
                    resume["iter"],
                    resume["performance"]["mean_ap"].item(),
                    resume["performance"]["mean_ap_50"].item(),
                    utils.to_json_str(ap)
                )
            except:
                logger.warning("No infomation from ckpt")
                if "performance" in resume:
                    logger.info(
                        "Old version performance: mean AP50: %.4f, AP50:\n%s",
                        resume["performance"]["map"],
                        utils.to_json_str(resume["performance"]["ap"])
                    )
            resume = resume["model"]
            resume = utils.convert_state_dict(resume)
            model.load_state_dict(resume)

    logger.info("Loading checkpoint Done")
    # load checkpoint
    model.to(device)
    if distributed:
        if train_cfg.get("sync_bn", False):
            logger.warning("Convert model `BatchNorm` to `SyncBatchNorm`")
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id])

    loss = build_set_criterion(loss_cfg, n_classes)
    loss.to(device)

    evaluator = get_evaluator(
        data_cfg["name"],
        loss,
        val_loader,
        loss_cfg["weight_dict"],
        device
    )

    evaluator = Evaluator(
        val_loader=val_loader,
        model=model,
        loss=loss,
        evaluator=evaluator,
        distributed=distributed,
        device=device
    )
    # start evaluation
    evaluator()
