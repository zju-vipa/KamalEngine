from light_detr.models.detr import DETR
import os
import logging
import shutil
import time
from logging.handlers import QueueHandler
from typing import Dict, Any, List, Set, Tuple
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
import light_detr.utils as detr_utils
from light_detr.eval import EvaluationBase, get_evaluator
from light_detr.data import build_train_dataset
from amalgamation.loss import get_amg_losses, AmalgamationLoss
from amalgamation.decay_strategy import DecayStrategy

from kamal.core.engine.engine import Engine


class Amalgamator(Engine):
    def __init__(self, logger=None, tb_writer=None):
        super(Amalgamator, self).__init__(logger=logger, tb_writer=tb_writer)

    def setup(
        self,
        train_cfg: Dict[str, Any],
        loss_cfg: Dict[str, Any],
        log_args: detr_utils.LogArgs,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler,
        student: nn.Module,
        teachers: List[nn.Module],
        loss: AmalgamationLoss,
        evaluator: EvaluationBase,
        distributed: bool,
        device: torch.device,
        resume: str = "",
        use_amp: bool = False
    ):
        # set up logger
        self._logger = logging.getLogger("amalgamator_rank_{}".format(dist_utils.get_rank()))

        # only write in master process
        self._tb_writer = None
        if dist_utils.is_main_process():
            self._tb_writer, _ = cv_utils.get_tb_writer(log_args.logdir, log_args.filename)
        dist_utils.barrier()

        self.train_cfg = train_cfg
        self.loss_cfg = loss_cfg
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
        self.student = student
        self.teachers = teachers
        self.loss = loss
        self.loss_weights = self.loss_cfg["weight_dict"]
        self.task_weights = self.loss_cfg["task_weights"]
        self.evaluator = evaluator
        self.distributed = distributed
        self.device = device
        self.ckpt_path = log_args.ckpt_path
        self.amp = use_amp
        # best index
        self.best_mean_ap = 0
        self.best_iter = 0
        # decay strategy
        use_decay = self.train_cfg.get("decay_strategy", False)
        self.decay_task = self.train_cfg.get("decay_task", True)
        decay_epoch = self.train_cfg.get("decay_epoch", self.total_epoch)
        self.decay_strategy = DecayStrategy(decay_epoch, use_decay)
        if use_decay:
            self._logger.info("With decay strategy")
        # for pytorch amp
        self.scaler: amp.GradScaler = None
        if self.amp:
            self._logger.info("Using AMP training")
            self.scaler = amp.GradScaler()
        self.resume(resume)

        # set extractor
        extract_layers = self.loss_cfg.get("mid_layers", list())
        self._logger.info("Extract names:\n%s", cv_utils.to_json_str(extract_layers))
        self.student_extractor = cv_utils.MidExtractor(self.student, extract_layers)
        self.teacher_extractors = list(cv_utils.MidExtractor(t, extract_layers) for t in self.teachers)
        self._logger.info("Start training for %d epochs", self.train_cfg["train_epochs"] - self.start_epoch)

    def resume(self, resume_fp: str = ""):
        """
        Resume training from checkpoint
        """
        # not a valid file
        if not os.path.isfile(resume_fp):
            return
        ckpt = torch.load(resume_fp, map_location="cpu")

        if isinstance(self.student, nn.parallel.DistributedDataParallel):
            self.student.module.load_state_dict(ckpt["model"])
        else:
            self.student.load_state_dict(ckpt["model"])
        if isinstance(self.loss, nn.parallel.DistributedDataParallel):
            self.loss.module.load_state_dict(ckpt["loss"], strict=False)
        else:
            self.loss.load_state_dict(ckpt["loss"], strict=False)

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
    def _check_nan_output(output, skip: List[Tensor]):
        pred_logits = output["pred_logits"]
        pred_boxes = output["pred_boxes"]
        if pred_logits.isnan().any() or pred_boxes.isnan().any():
            skip[dist_utils.get_rank()].fill_(True)

    def train_iter(self, x: detr_utils.NestedTensor, targets: List[Dict[str, Any]]):
        self.student.train()
        self.loss.train()
        # move to device
        x, targets = detr_utils.move_data_to_device(x, targets, self.device)

        self.optimizer.zero_grad()
        try:
            with amp.autocast(enabled=self.amp):
                skip = list(torch.tensor(False, device=self.device) for _ in range(dist_utils.get_world_size()))
                # teachers are already no grad
                output_t = list(t(x) for t in self.teachers)
                feat_t = list(t_f.features for t_f in self.teacher_extractors)
                # get info from teachers
                drop_info = dict(
                    epoch=self.epoch,
                    total_epoch=self.total_epoch,
                    feat_t=list(t["feat"].detach() for t in output_t)
                )
                output_s = self.student(x, drop_info=drop_info)
                feat_s = self.student_extractor.features
                self._check_nan_output(output_s, skip)
                skip = self._collect_all_skip(skip)
                if skip.any():
                    self._logger.info(f"Got skip: {skip}, skiping this iter")
                    return
                loss_dict: Dict[str, torch.Tensor] = self.loss(
                    output_s=output_s,
                    output_t=output_t,
                    target=targets,
                    student_seq=feat_s,
                    teacher_seq=feat_t
                )
                weighted_loss_dict: Dict[str, torch.Tensor] = dict()
                for k, loss in list(loss_dict.items()):
                    k_prefix = k.split(".")[0]
                    if k_prefix in self.loss_weights:
                        task_w = self.task_weights["task"]
                        amg_w = self.task_weights["amg"]
                        if self.decay_strategy.enable:
                            decay_amg_w = self.decay_strategy(self.epoch)
                            amg_w = amg_w * decay_amg_w
                            if self.decay_task:
                                task_w = (1 - decay_amg_w) * task_w
                        if "amg" in k:
                            weight = amg_w
                        else:
                            weight = task_w
                        weighted_loss_dict[k] = loss * self.loss_weights[k_prefix] * weight
                    else:
                        loss_dict.pop(k)
                loss: torch.Tensor = sum(weighted_loss_dict.values())
        # critical error
        except detr_utils.ErrorBBOX as e:
            os.makedirs("debug", exist_ok=True)
            fp = "debug/error_bbox.pth"
            self._logger.error(f"bbox is error: {e.bbox}, store in {fp}")
            state_dict = {
                "student": self.student.state_dict(),
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
                "student": self.student.state_dict(),
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
                    self.student.parameters(),
                    self.train_cfg["clip_max_norm"]
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            # grad clip
            if "clip_max_norm" in self.train_cfg:
                nn.utils.clip_grad.clip_grad_norm_(
                    self.student.parameters(),
                    self.train_cfg["clip_max_norm"]
                )
            self.optimizer.step()

        # save memory
        self.optimizer.zero_grad(set_to_none=True)

        weighted_loss: torch.Tensor = dist_utils.reduce_tensor(loss.detach())
        loss_dict = dist_utils.reduce_dict(loss_dict)
        weighted_loss_dict = dist_utils.reduce_dict(weighted_loss_dict)
        # print
        if self.iter % self.train_cfg["print_interval"] == 0 and dist_utils.is_main_process():
            for k, v in loss_dict.items():
                loss_dict[k] = round(v.item(), 4)
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
            self._tb_writer.add_scalars("Train/Weighted_loss_dict", weighted_loss_dict, self.iter)
            self._tb_writer.add_scalars("Train/Error_dict", error_dict, self.iter)
            self._tb_writer.add_scalar("DecayStrategy", self.decay_strategy(self.epoch), self.iter)
        dist_utils.barrier()
        self.iter += 1

    def validate_and_save(self):
        self._logger.info("Start evaluation")
        ret_dict = self.evaluator(self.student)
        if isinstance(self.student, nn.parallel.DistributedDataParallel):
            model_state_dict = self.student.module.state_dict()
        else:
            model_state_dict = self.student.state_dict()
        if isinstance(self.loss, nn.parallel.DistributedDataParallel):
            loss_state_dict = self.loss.module.state_dict()
        else:
            loss_state_dict = self.loss.state_dict()

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
                "loss": loss_state_dict,
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
            self._logger.info("Saving state dict...")
            torch.save(state_dict, os.path.join(self.ckpt_path, f"iter-{self.iter}.pth"))
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
        self._logger.info("Training time %s", total_time_str)


def build_models(
    student_cfg: Dict[str, Any],
    teachers_cfg: List[Dict[str, Any]],
    device: torch.device
) -> Tuple[List[DETR], DETR, List[int], List[List[int]]]:
    """
    Build multitask teachers and student model
    """
    logger = logging.getLogger("build_models")
    useless_teacher_keys = [
        "training",
        "validation",
        "loss"
    ]

    teachers: List[DETR] = list()
    teacher_tasks: List[Set[int]] = list()
    # build teachers
    for teacher_name in sorted(teachers_cfg):
        teacher_cfg = cv_utils.get_cfg(teachers_cfg[teacher_name]["cfg_fp"])
        for k in useless_teacher_keys:
            teacher_cfg.pop(k)
        logger.info("Building teacher %s with configure:\n%s", teacher_name, yaml.dump(teacher_cfg))
        # get teacher task
        teacher_task = set(teacher_cfg["dataset"]["make_partial"])
        teacher_tasks.append(teacher_task)
        # get teacher model
        teacher = build_detr(teacher_cfg["model"], num_classes=len(teacher_task) + 1)
        # load teacher pretrained weight
        teacher_weight_fp = teachers_cfg[teacher_name]["weights_fp"]
        if teacher_weight_fp is not None:
            ckpt = torch.load(teacher_weight_fp, map_location="cpu")
            teacher.load_state_dict(ckpt["model"])
        # set not learnable for both parameters and bn state
        teacher.requires_grad_(False)
        teacher.eval()
        # move to device
        teacher.to(device)
        teachers.append(teacher)
    # check teacher tasks
    student_classes = set()
    for t_task in teacher_tasks:
        assert not (student_classes & t_task), f"teacher has common classes {student_classes & t_task}"
        student_classes |= t_task
    # build student
    student_classes = sorted(list(student_classes))
    teacher_tasks = list(sorted(list(t)) for t in teacher_tasks)
    logger.info(
        "Building detr student with %d classes:\n %s",
        len(student_classes),
        cv_utils.to_json_str(student_classes)
    )
    student = build_detr(student_cfg, len(student_classes) + 1)
    logger.info(
        "Built student with %d parameters, %d trainable parameters",
        cv_utils.count_parameters(student, include_no_grad=True),
        cv_utils.count_parameters(student, include_no_grad=False)
    )
    if student_cfg.get("pre_train", None) is not None:
        detr_utils.load_pretrain_model(
            student_cfg["pre_train"],
            student,
            num_proj=student_cfg["detr"].get("num_proj", 1)
        )
        logger.info("Loaded pretrain model: %s", student_cfg["pre_train"])
    student.to(device)
    return teachers, student, student_classes, teacher_tasks


def amalgamate_worker(
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
    teachers_cfg: Dict[str, Any] = global_cfg["teachers"]
    student_cfg: Dict[str, Any] = global_cfg["student"]
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

    # set cuda
    logger.info("Use GPU: %d for training", gpu_id)
    device = torch.device("cuda:{}".format(gpu_id))
    # IMPORTANT! for distributed training (reduce_all_object)
    torch.cuda.set_device(device)

    if current_rank == 0:
        logger.info("Starting with configs:\n%s", yaml.dump(global_cfg))

    # make determinstic
    if launch_args.seed is not None:
        seed = launch_args.seed + current_rank
        logger.info("Initial rank %d with seed: %d", current_rank, seed)
        cv_utils.make_deterministic(seed)
    torch.backends.cudnn.benchmark = True

    # get dataloader
    logger.info("Building dataset...")
    train_loader, val_loader, n_classes = build_train_dataset( 
        data_cfg,
        train_cfg,
        val_cfg,
        launch_args,
    ) # train and val

    # create model
    teachers, student, student_classes, teacher_classes = build_models(
        student_cfg,
        teachers_cfg,
        device,
    ) # teacher model(load_state_dict) and student model 
    assert n_classes == len(student_classes) + 1, "student must be trained on full task"

    def make_distributed_model(model: nn.Module):
        if train_cfg.get("sync_bn", False):
            logger.warning("Convert model `BatchNorm` to `SyncBatchNorm`")
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device.index])
        return model

    if distributed:
        # teacher do not need distributed
        student = make_distributed_model(student)

    param_dicts = [
        {
            "params": [p for n, p in student.named_parameters() if "backbone" not in n and p.requires_grad]
        },
        {
            "params": [p for n, p in student.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": train_cfg["lr_backbone"],
        },
    ]
    optimizer = get_optimizer(param_dicts, train_cfg["optimizer"])
    logger.info("Loaded optimizer:\n%s", optimizer) # optimizer parameter
    lr_scheduler = get_scheduler(optimizer, train_cfg["lr_schedule"])

    # KAE Part
    amalgamator = Amalgamator()

    train_loss = get_amg_losses(
        loss_cfg=loss_cfg,
        teacher_tasks=teacher_classes,
        total_tasks=student_classes
    )
    train_loss.to(device)
    if distributed and cv_utils.count_parameters(train_loss) != 0:
        train_loss = make_distributed_model(train_loss)
        eval_loss = train_loss.module.task_loss
    else:
        eval_loss = train_loss.task_loss

    evaluator = get_evaluator(
        data_cfg["name"],
        eval_loss,
        val_loader,
        loss_cfg["weight_dict"],
        device
    )

    amalgamator.setup(
        train_cfg=train_cfg,
        loss_cfg=loss_cfg,
        log_args=log_args,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        student=student,
        teachers=teachers,
        loss=train_loss,
        evaluator=evaluator,
        distributed=distributed,
        device=device,
        resume=resume,
        use_amp=launch_args.use_amp
    )
    # start training
    amalgamator.run()
