import random
import argparse
import logging
from typing import Optional
import threading
import os
from typing import Dict, Any
import numpy as np

import torch
from torch.optim import Optimizer


def adjust_learning_rate_new(epoch, optimizer, LUT):
    """
    new learning rate schedule according to RotNet
    """
    lr = next((lr for (max_epoch, lr) in LUT if max_epoch > epoch), LUT[-1][1])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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

def adjust_learning_rate_stage_agent(
        optimizer: Optimizer,
        cfg: Dict[str, Any],
        epoch: int):

    if epoch<30:
        for param_group in optimizer.param_groups:
            param_group['lr'] = cfg["training"]["lr_agent"]/100
        return

    steps = np.sum(epoch > np.asarray(cfg["training"]["lr_decay_epochs"]))
    if steps > 0:
        new_lr = cfg["training"]["lr_agent"] * (cfg["training"]["lr_decay_rate"] ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

def adjust_learning_rate_stage(
        optimizer: Optimizer,
        cfg: Dict[str, Any],
        epoch: int):

    steps = np.sum(epoch > np.asarray(cfg["training"]["lr_decay_epochs"]))

    if steps > 0:
        new_lr = cfg["training"]["optimizer"]["lr"] * (cfg["training"]["lr_decay_rate"] ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

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