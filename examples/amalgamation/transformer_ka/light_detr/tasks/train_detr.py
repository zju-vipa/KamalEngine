import os
import logging
import shutil
import time
from logging.handlers import QueueHandler
from typing import Dict, Any, List
import datetime
import yaml

import torch
from torch import nn, Tensor
import torch.cuda
import torch.distributed as dist
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn
from torch.cuda import amp

import cv_lib.utils as cv_utils
from cv_lib.optimizers import get_optimizer
from cv_lib.schedulers import get_scheduler
import cv_lib.distributed.utils as dist_utils

from light_detr.models import build_detr
from light_detr.loss import build_set_criterion, SetCriterion
import light_detr.utils as detr_utils
from light_detr.eval import EvaluationBase, get_evaluator
from light_detr.data import build_train_dataset

from kamal.core.engine.engine import Engine


class Trainer(Engine):
    def __init__(self, logger=None, tb_writer=None):
        super(Trainer, self).__init__(logger=logger, tb_writer=tb_writer)
        
    def setup(
        self,
        train_cfg: Dict[str, Any],
        log_args: detr_utils.LogArgs,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler,
        model: nn.Module,
        loss: SetCriterion,
        loss_weights: Dict[str, float],
        evaluator: EvaluationBase,
        distributed: bool,
        device: torch.device,
        resume: str = "",
        use_amp: bool = False
    ):
        # set up logger
        self._logger = logging.getLogger("trainer_rank_{}".format(dist_utils.get_rank()))

        # only write in master process
        self._tb_writer = None
        if dist_utils.is_main_process():
            self._tb_writer, _ = cv_utils.get_tb_writer(log_args.logdir, log_args.filename)
        dist_utils.barrier()

        self.train_cfg = train_cfg
        self.start_epoch = 0
        self.epoch = 0
        self.total_epoch = self.train_cfg["train_epochs"]
        self.iter = 0
        self.step = 0
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.total_step = len(self.train_loader)
        self.total_iter = self.total_step * self.total_epoch
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.model = model
        self.loss = loss
        self.loss_weights = loss_weights
        self.evaluator = evaluator
        self.distributed = distributed
        self.device = device
        self.ckpt_path = log_args.ckpt_path
        self.amp = use_amp
        # best index
        self.best_mean_ap = 0
        self.best_iter = 0

        # for pytorch amp
        self.scaler: amp.GradScaler = None
        if self.amp:
            self._logger.info("Using AMP train")
            self.scaler = amp.GradScaler()
        self.resume(resume)
        self._logger.info("Start training for %d epochs", self.train_cfg["train_epochs"] - self.start_epoch)

    def resume(self, resume_fp: str = ""):
        """
        Resume training from checkpoint
        """
        # not a valid file
        if not os.path.isfile(resume_fp):
            return
        ckpt = torch.load(resume_fp, map_location="cpu")

        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            self.model.module.load_state_dict(ckpt["model"])
        else:
            self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler_statedict = {
            "last_epoch": ckpt["lr_scheduler"]["last_epoch"],
            "_step_count": ckpt["lr_scheduler"]["_step_count"],
            "_last_lr": ckpt["lr_scheduler"]["_last_lr"]
        }
        self.lr_scheduler.load_state_dict(lr_scheduler_statedict)
        # load grad scaler
        if self.scaler is not None and "grad_scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["grad_scaler"])
        self.iter = ckpt["iter"] + 1
        self.start_epoch = ckpt["epoch"] + 1
        self._logger.info("Loaded ckpt with epoch: %d, iter: %d", ckpt["epoch"], ckpt["iter"])

    @staticmethod
    def _collect_all_skip(skip: List[Tensor]) -> Tensor:
        dist_utils.barrier()
        for i in range(dist_utils.get_world_size()):
            dist_utils.broadcast_tensor(skip[i], i)
        dist_utils.barrier()
        skip = torch.stack(skip)
        return skip

    @staticmethod
    def _check_nan_output(output: Dict[str, Tensor], skip: List[Tensor]):
        pred_logits = output["pred_logits"]
        pred_boxes = output["pred_boxes"]
        if pred_logits.isnan().any() or pred_boxes.isnan().any():
            skip[dist_utils.get_rank()].fill_(True)

    def train_iter(self, x: detr_utils.NestedTensor, targets: List[Dict[str, Any]]):
        self.model.train()
        self.loss.train()
        # move to device
        x, targets = detr_utils.move_data_to_device(x, targets, self.device)

        self.optimizer.zero_grad()
        try:
            with amp.autocast(enabled=self.amp):
                skip = list(torch.tensor(False, device=self.device) for _ in range(dist_utils.get_world_size()))
                output = self.model(x)
                self._check_nan_output(output, skip)
                skip = self._collect_all_skip(skip)
                if skip.any():
                    self._logger.info(f"Got skip: {skip}, skiping this iter")
                    return
                loss_dict: Dict[str, torch.Tensor] = self.loss(output, targets)
                weighted_loss: Dict[str, torch.Tensor] = dict()
                for k, loss in loss_dict.items():
                    k_prefix = k.split(".")[0]
                    if k_prefix in self.loss_weights:
                        weighted_loss[k] = loss * self.loss_weights[k_prefix]
                loss: torch.Tensor = sum(weighted_loss.values())
        # critical error
        except detr_utils.ErrorBBOX as e:
            os.makedirs("debug", exist_ok=True)
            fp = "debug/error_bbox.pth"
            self._logger.error(f"bbox is error: {e.bbox}, store in {fp}")
            state_dict = {
                "model": self.model.state_dict(),
                "epoch": self.epoch,
                "iter": self.iter,
                "bbox": e.bbox,
                "x": x.tensors,
                "mask": x.mask,
                "targets": targets
            }
            torch.save(state_dict, fp)
            raise
        # unknown error
        except Exception:
            os.makedirs("debug", exist_ok=True)
            fp = "debug/unknown_error.pth"
            self._logger.error(f"unknown error, store in {fp}")
            state_dict = {
                "model": self.model.state_dict(),
                "epoch": self.epoch,
                "iter": self.iter,
                "x": x.tensors,
                "mask": x.mask,
                "targets": targets
            }
            torch.save(state_dict, fp)
            raise

        if self.amp:
            self.scaler.scale(loss).backward()
            # grad clip
            if "clip_max_norm" in self.train_cfg:
                # Un-scales the gradients of optimizer's assigned params in-place
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_cfg["clip_max_norm"]
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            # grad clip
            if "clip_max_norm" in self.train_cfg:
                nn.utils.clip_grad.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_cfg["clip_max_norm"]
                )
            self.optimizer.step()

        # save memory
        self.optimizer.zero_grad(set_to_none=True)

        weighted_loss: torch.Tensor = dist_utils.reduce_tensor(loss.detach())
        loss_dict = dist_utils.reduce_dict(loss_dict)
        # print
        if self.iter % self.train_cfg["print_interval"] == 0 and dist_utils.is_main_process():
            loss_dict = cv_utils.tensor_dict_items(loss_dict, ndigits=4)
            # reduce loss
            self._logger.info(
                "Epoch %3d|%3d, step %4d|%4d, iter %5d|%5d, lr:\n%s,\nloss: %.5f, loss dict: %s",
                self.epoch, self.total_epoch,
                self.step, self.total_step,
                self.iter, self.total_iter,
                cv_utils.to_json_str(self.lr_scheduler.get_last_lr()),
                weighted_loss.item(),
                cv_utils.to_json_str(loss_dict)
            )
            self._tb_writer.add_scalar("Loss/Train", weighted_loss, self.iter)
            error_dict = dict()
            for k in list(loss_dict.keys()):
                if "error" in k:
                    error_dict[k] = loss_dict.pop(k)
            self._tb_writer.add_scalars("Train/Loss_dict", loss_dict, self.iter)
            self._tb_writer.add_scalars("Train/Error_dict", error_dict, self.iter)
        dist_utils.barrier()
        self.iter += 1

    def validate_and_save(self):
        self._logger.info("Start evaluation")
        ret_dict = self.evaluator(self.model)
        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        if dist_utils.is_main_process():
            self._logger.info("evaluation done")
            loss = ret_dict["loss"]
            loss_dict = ret_dict["loss_dict"]
            loss_dict = cv_utils.tensor_dict_items(loss_dict, ndigits=4)
            performance: Dict[str, torch.Tensor] = ret_dict["performance"]
            mean_ap = performance["mean_ap"].item()
            mean_ap_50 = performance["mean_ap_50"].item()
            mean_ap_75 = performance["mean_ap_75"].item()
            # write logger
            info = "Validation loss: {:.5f}, mean AP: {:.4f}, mean AP50: {:.4f}, mean AP75: {:.4f}\nloss dict: {}\nAP: {}"
            info = info.format(
                loss,
                mean_ap,
                mean_ap_50,
                mean_ap_75,
                cv_utils.to_json_str(loss_dict),
                cv_utils.to_json_str(cv_utils.tensor_to_list(performance["ap"]))
            )
            self._logger.info(info)
            # write tb logger
            self._tb_writer.add_scalar("Loss/Val", loss, self.iter)
            self._tb_writer.add_scalar("Val/Mean AP-50", mean_ap_50, self.iter)
            self._tb_writer.add_scalar("Val/Mean AP-75", mean_ap_75, self.iter)
            self._tb_writer.add_scalar("Val/Mean AP", mean_ap, self.iter)
            error_dict = dict()
            for k in list(loss_dict.keys()):
                if "error" in k:
                    error_dict[k] = loss_dict.pop(k)
            self._tb_writer.add_scalars("Val/Loss_dict", loss_dict, self.iter)
            self._tb_writer.add_scalars("Val/Error_dict", error_dict, self.iter)
            label_info = self.val_loader.dataset.label_info
            for cls_id, ap in enumerate(performance["ap"]):
                # ignore background
                if cls_id == 0:
                    continue
                self._tb_writer.add_scalar(f"Val/AP/{label_info[cls_id]}", ap, self.iter)

            # save ckpt
            state_dict = {
                "model": model_state_dict,
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "epoch": self.epoch,
                "iter": self.iter,
                "performance": performance,
                "loss_dict": loss_dict,
            }
            save_fp = os.path.join(self.ckpt_path, f"iter-{self.iter}.pth")
            self._logger.info("Saving state dict to %s...", save_fp)
            torch.save(state_dict, save_fp)
            if mean_ap > self.best_mean_ap:
                # best index
                self.best_mean_ap = mean_ap
                self.best_iter = self.iter
                shutil.copy(save_fp, os.path.join(self.ckpt_path, "best.pth"))
        dist_utils.barrier()

    def run(self):
        start_time = time.time()
        # start one epoch
        for self.epoch in range(self.start_epoch, self.train_cfg["train_epochs"]):
            if self.distributed:
                self.train_loader.sampler.set_epoch(self.epoch)
            for self.step, (x, target) in enumerate(self.train_loader):
                self.train_iter(x, target)
                # validation
                if self.iter % self.train_cfg["val_interval"] == 0:
                    self.validate_and_save()
            self.lr_scheduler.step()
        self._logger.info("Final validation")
        self.validate_and_save()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        if dist_utils.is_main_process():
            self._logger.info("Training time %s", total_time_str)
            self._logger.info("Best mean ap: %f, iter: %d", self.best_mean_ap, self.best_iter)
        dist_utils.barrier()


def train_worker(
    gpu_id: int,
    launch_args: detr_utils.DistLaunchArgs,
    log_args: detr_utils.LogArgs,
    global_cfg: Dict[str, Any],
    resume: str = ""
):
    """
    What created in this function is only used in this process and not shareable
    """
    # setup process root logger
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
    model_cfg: Dict[str, Any] = global_cfg["model"]
    loss_cfg: Dict[str, Any] = global_cfg["loss"]
    # set debug number of workers
    if launch_args.debug:
        train_cfg["num_workers"] = 0
        val_cfg["num_workers"] = 0
        train_cfg["print_interval"] = 1
        train_cfg["val_interval"] = 10
    distributed = launch_args.distributed
    # get current rank
    current_rank = launch_args.rank
    if distributed:
        if launch_args.multiprocessing:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
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

    if current_rank == 0:
        logger.info("Starting with configs:\n%s", yaml.dump(global_cfg))

    # make determinstic
    if launch_args.seed is not None:
        seed = launch_args.seed + current_rank
        logger.info("Initial rank %d with seed: %d", current_rank, seed)
        cv_utils.make_deterministic(seed)
    # set cuda
    torch.backends.cudnn.benchmark = True
    logger.info("Use GPU: %d for training", gpu_id)
    device = torch.device("cuda:{}".format(gpu_id))
    # IMPORTANT! for distributed training (reduce_all_object)
    torch.cuda.set_device(device)

    # get dataloader
    logger.info("Building dataset...")
    train_loader, val_loader, n_classes = build_train_dataset( 
        data_cfg,
        train_cfg,
        val_cfg,
        launch_args,
    )
    # create model
    logger.info("Building detr...")
    model = build_detr(model_cfg, n_classes, seg=False)
    logger.info(
        "Built model with %d parameters, %d trainable parameters",
        cv_utils.count_parameters(model, include_no_grad=True),
        cv_utils.count_parameters(model, include_no_grad=False)
    )
    if model_cfg.get("pre_train", None) is not None: # pre_train
        detr_utils.load_pretrain_model(model_cfg["pre_train"], model)
        logger.info("Loaded pretrain model: %s", model_cfg["pre_train"])
    model.to(device)
    if distributed:
        if train_cfg.get("sync_bn", False):
            logger.warning("Convert model `BatchNorm` to `SyncBatchNorm`")
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id])
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": train_cfg["lr_backbone"],
        },
    ]
    optimizer = get_optimizer(param_dicts, train_cfg["optimizer"])
    logger.info("Loaded optimizer:\n%s", optimizer)
    lr_scheduler = get_scheduler(optimizer, train_cfg["lr_schedule"])

    # KAE Part
    trainer = Trainer()

    loss = build_set_criterion(loss_cfg, n_classes)
    loss.to(device)

    evaluator = get_evaluator(
        data_cfg["name"],
        loss,
        val_loader,
        loss_cfg["weight_dict"],
        device
    )

    trainer.setup(
        train_cfg=train_cfg,
        log_args=log_args,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        model=model,
        loss=loss,
        loss_weights=loss_cfg["weight_dict"],
        evaluator=evaluator,
        distributed=distributed,
        device=device,
        resume=resume,
        use_amp=launch_args.use_amp
    )
    # start training
    trainer.run()
