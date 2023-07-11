import random
import argparse
import logging
from typing import Optional
import threading
import os

import numpy as np
import tqdm
import torch
from torch import Tensor
from torch.nn import Module, ModuleDict
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import Dataset,DataLoader
from torchvision.datasets import CIFAR100, CIFAR10
from torchvision import transforms
from typing import Dict, Any, Iterable, List
from torch.optim import *
import copy

from kamal.vision.models.classification.resnetv2 import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from kamal.vision.models.classification.wresnet import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2
from kamal.vision.models.classification.vgg import vgg19_bn, vgg16_bn, vgg13_bn, vgg11_bn
from kamal.vision.models.classification.mobilenetv2 import mobilenet_v2
from kamal.vision.models.classification.shufflenet import ShuffleV1
from kamal.vision.models.classification.shufflenetv2 import shuffle_v2

#choose model
MODEL_DICT = {
    'ResNet18': ResNet18,
    'ResNet34': ResNet34,
    'ResNet50': ResNet50,
    'ResNet101': ResNet101,
    'ResNet152': ResNet152,
    'wrn_16_1': wrn_16_1,
    'wrn_16_2': wrn_16_2,
    'wrn_40_1': wrn_40_1,
    'wrn_40_2': wrn_40_2,
    'vgg11': vgg11_bn,
    'vgg13': vgg13_bn,
    'vgg16': vgg16_bn,
    'vgg19': vgg19_bn,
    'MobileNetV2': mobilenet_v2,
    'ShuffleV1': ShuffleV1,
    'ShuffleV2': shuffle_v2,
}

def get_model(model_name: str, num_classes: int, state_dict: Dict[str, torch.Tensor] = None, **kwargs):
    fn = MODEL_DICT[model_name]
    model = fn(num_classes=num_classes, **kwargs)

    if state_dict is not None:
        model.load_state_dict(state_dict)
    return model

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


def get_cifar_10(root: str, split: str = "train") -> Dataset:
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    if split == "train":
        transform = train_transform
        is_train = True
    else:
        transform = test_transform
        is_train = False
    
    # if loss_method == 'nce':
    #     target_transform = convert_one_hot_cifar10
    # elif loss_method == 'ce':
    #     target_transform = None

    target_transform = None

    dataset = CIFAR10(
        root=root,
        train=is_train,
        transform=transform,
        target_transform=target_transform,
        download=True
    )

    return dataset

def get_cifar_100(root: str,  split: str = "train") -> Dataset:
    normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    if split == "train":
        transform = train_transform
        is_train = True
    else:
        transform = test_transform
        is_train = False

    # if loss_method == 'nce':
    #     target_transform = convert_one_hot_cifar100
    # elif loss_method == 'ce':
    #     target_transform = None

    target_transform = None

    dataset = CIFAR100(
        root=root,
        train=is_train,
        transform=transform,
        target_transform=target_transform,
        download=True
    )

    return dataset

DATASET_DICT = {
    "cifar10": get_cifar_10,
    "cifar100": get_cifar_100,
    "CIFAR10": get_cifar_10,
    "CIFAR100": get_cifar_100,
    # "imagenet": get_imagenet,
    # "tiny-imagenet": get_tiny_imagenet
}

def get_dataset(name: str, root: str, split: str = "train", **kwargs) -> Dataset:
    fn = DATASET_DICT[name]
    return fn(root=root, split=split)

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



OptimizerDict = dict(
    SGD=SGD,
    Adadelta=Adadelta,
    Adagrad=Adagrad,
    Adam=Adam,
    AdamW=AdamW,
    SparseAdam=SparseAdam,
    Adamax=Adamax,
    ASGD=ASGD,
    Rprop=Rprop,
    RMSprop=RMSprop,
    LBFGS=LBFGS
)

def get_optimizer(params: Iterable, optim_cfg: Dict[str, Any]) -> Optimizer:
    name = optim_cfg["name"]
    optimizer = OptimizerDict[name]

    kwargs = copy.deepcopy(optim_cfg)
    kwargs.pop("name")

    return optimizer(params=params, **kwargs)

def adjust_learning_rate_stage2(
    optimizer: Optimizer,
    epoch_current: int):
    
    global_lr = optimizer.param_groups[0]['lr']

    # if epoch_current >= 10 and epoch_current % 10 == 0:
    #     global_lr = global_lr/2

    if epoch_current > 0 and epoch_current % 30 == 0:
        global_lr = global_lr/2

    optimizer.param_groups[0]['lr'] = global_lr
    return global_lr

def adjust_learning_rate(
    cfg_lr_global: float,
    cfg_lr_branch: float,
    optimizer: Optimizer,
    epoch_current: int,
    epoch_sum: int
    ):

    global_lr = optimizer.param_groups[0]['lr']
    branch_lr = optimizer.param_groups[1]['lr']
    # print(global_lr, branch_lr, type(global_lr))

    # warmup without update branch probabilities
    if epoch_current-1 == 0:
        # print(epoch_current-1, cfg_lr_global, type(cfg_lr_global))
        global_lr = cfg_lr_global / 4
        branch_lr = 0
    elif epoch_current-1 == 1:
        global_lr = cfg_lr_global / 2
        branch_lr = 0
    elif epoch_current-1 == 2:
        global_lr = cfg_lr_global
        branch_lr = cfg_lr_branch

    # exponential decay 2.4
    elif (epoch_current-1) % 2 == 0:
        global_lr = cfg_lr_global * (1 - (epoch_current-1) / int(epoch_sum * 1.03093))
        branch_lr = cfg_lr_branch * (1 - (epoch_current-1) / int(epoch_sum * 1.03093))

    optimizer.param_groups[0]['lr'] = global_lr
    optimizer.param_groups[1]['lr'] = branch_lr
    return global_lr, branch_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1 / batch_size))
        return res


def make_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def str2bool(v):
    if v.lower() in ("true", "yes", "t", "y"):
        return True
    elif v.lower() in ("false", "no", "f", "n"):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")


def str2loglevel(v):
    if v.lower() in ("debug", "d", "10"):
        return logging.DEBUG
    elif v.lower() in ("info", "i", "20"):
        return logging.INFO
    elif v.lower() in ("warning", "warn", "w", "30"):
        return logging.WARNING
    elif v.lower() in ("error", "e", "40"):
        return logging.ERROR
    elif v.lower() in ("fatal", "critical", "50"):
        return logging.FATAL
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")


def get_logger(
    level: int,
    logger_fp: str,
    name: Optional[str] = None,
    mode: str = "w",
    format: str = "%(asctime)s - %(funcName)s - %(levelname)s - %(message)s"
):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    file_handler = logging.FileHandler(logger_fp, "w")
    file_handler.setLevel(level)
    formatter = logging.Formatter(format)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger

def preserve_gpu_with_id(gpu_id: int, preserve_percent: float = 0.95):
    logger = logging.getLogger("preserve_gpu_with_id")
    if not torch.cuda.is_available():
        logger.warning("no gpu avaliable exit...")
        return
    try:
        import cupy
        device = cupy.cuda.Device(gpu_id)
        avaliable_mem = device.mem_info[0] - 700 * 1024 * 1024
        logger.info("{}MB memory avaliable, trying to preserve {}MB...".format(
            int(avaliable_mem / 1024.0 / 1024.0),
            int(avaliable_mem / 1024.0 / 1024.0 * preserve_percent)
        ))
        if avaliable_mem / 1024.0 / 1024.0 < 100:
            cmd = os.popen("nvidia-smi")
            outputs = cmd.read()
            pid = os.getpid()

            logger.warning("Avaliable memory is less than 100MB, skiping...")
            logger.info("program pid: %d, current environment:\n%s", pid, outputs)
            raise Exception("Memory Not Enough")
        alloc_mem = int(avaliable_mem * preserve_percent / 4)
        x = torch.empty(alloc_mem).to(torch.device("cuda:{}".format(gpu_id)))
        del x
    except ImportError:
        logger.warning("No cupy found, memory cannot be perserved")


def preserve_memory(preserve_percent: float = 0.99):
    logger = logging.getLogger("preserve_memory")
    if not torch.cuda.is_available():
        logger.warning("no gpu avaliable exit...")
        return
    thread_pool = list()
    for i in range(torch.cuda.device_count()):
        thread = threading.Thread(
            target=preserve_gpu_with_id,
            kwargs=dict(
                gpu_id=i,
                preserve_percent=preserve_percent
            ),
            name="Preserving GPU {}".format(i)
        )
        logger.info("Starting to preserve GPU: {}".format(i))
        thread.start()
        thread_pool.append(thread)
    for t in thread_pool:
        t.join()

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


def validate_LTB(
    cfg: Dict[str, Any],
    val_loader: DataLoader,
    model: Module,
    criterion: Module,
    device: torch.device,
    num_classes: int,
    t: int,
    epoch: int,
    loss_method: str,
    stage: str    
    ):

    logger = logging.getLogger("validate")
    logger.info("Start validation")

    # if loss_method == 'nce':
    if cfg["model"]["task"] == 'mt':

        losses = [AverageMeter() for _ in range(num_classes)]
        top1 = [AverageMeter() for _ in range(num_classes)]
        top5 = [AverageMeter() for _ in range(num_classes)]

    # elif loss_method =='ce':
    elif cfg["model"]["task"] == 'mc':

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

    model.eval()

    with torch.no_grad():
        for x, target in tqdm.tqdm(val_loader):
            x = x.to(device)
            target = target.to(device)

            if stage == 's1':
                output = model(x, t/(epoch), True)
            elif stage == 's2':
                output = model(x, t/(epoch), False)

            # if loss_method == 'ce':
            if cfg["model"]["task"] == 'mc':

                loss = criterion(output, target.squeeze())
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
             
                # prec1 = accuracy(output, target.squeeze())
                losses.update(loss.item(), x.shape[0])
                top1.update(acc1[0], x.shape[0])
                top5.update(acc5[0], x.shape[0])   

                loss_avg = losses.avg
                top1_avg = top1.avg
                top5_avg = top5.avg

            # elif loss_method == 'nce':
            elif cfg["model"]["task"] == 'mt':

                loss = []
                acc1, acc5 = [], []

                for j in range(len(output)):
                    loss.append(criterion(output[j], target[:, j]))
                    acc1.append(accuracy(output[j], target[:, j], topk=(1, 1))[0])
                    acc5.append(accuracy(output[j], target[:, j], topk=(1, 1))[1])

                    losses[j].update(loss[j].item(), x.shape[0])
                    top1[j].update(acc1[j], x.shape[0])
                    top5[j].update(acc5[j], x.shape[0])

                losses_avg = [losses[k].avg for k in range(len(losses))]
                top1_avg = [top1[k].avg for k in range(len(top1))]
                top5_avg = [top5[k].avg for k in range(len(top5))]

                loss_avg = sum(losses_avg) / len(losses_avg)
                top1_avg = sum(top1_avg) / len(top1_avg)
                top5_avg = sum(top5_avg) / len(top5_avg)

                # loss = sum(loss)
        logger.info("acc@1: %.4f acc@5: %.4f, loss: %.5f", top1_avg, top5_avg, loss_avg)
        # logger.info("acc@1: %.4f acc@5: %.4f, loss: %.5f", top1.avg, top5.avg, losses.avg)

    # return top1.avg, top5.avg, losses.avg
    return top1_avg, top5_avg, loss_avg
    # return (loss_avg, prec1_avg)

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
