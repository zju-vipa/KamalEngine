import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
import argparse
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms as T
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn import Module, CrossEntropyLoss, ModuleDict

from visdom import Visdom
from kamal import engine, slim, callbacks,utils
import logging

import yaml


from utils import get_cifar_10, get_cifar_100
from loss import get_loss_module, get_loss_forward

from utils import str2bool, preserve_memory, get_logger, get_model, get_teacher, get_dataloader, get_optimizer
from utils import make_deterministic
from utils import AverageMeter, accuracy,validate
from LTBkd import LTBDistiller
from kamal.vision.models.classification.LEARNTOBRANCH import LEARNTOBRANCH_Deep
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default = './config/kd/cifar10/ResNet50-LTB-ce.yml')
    parser.add_argument("--logdir", type=str, default = './log_LTB_kd_c10_ce')
    parser.add_argument("--file_name_cfg", type=str,default = 'ResNet50-LTB-ce.yml')
    parser.add_argument("--stage", type=str,default = 's1')
    parser.add_argument("--gpu_preserve", type=str2bool, default=False)
    parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument("--preserve_percent", type=float, default=0.95)
    args = parser.parse_args()

    __global_values__ = dict(it=0)

    #get config
    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.SafeLoader)
    
    # make dirs
    args.logdir = args.logdir+'_'+args.stage
    ckpt_dir = os.path.join(args.logdir, "ckpt")
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    train_log_dir = os.path.join(args.logdir, "train-logs")
    os.makedirs(train_log_dir, exist_ok=True)

    # get logger
    formatter = (
        cfg["kd"]["teacher"]["name"],
        cfg["kd"]["student"]["name"],
        cfg["kd_loss"]["name"],
        cfg["dataset"]["name"],
    )
    logger = get_logger(
        level=logging.INFO,
        mode="w",
        name=None,
        logger_fp=os.path.join(
            train_log_dir,
            "training-" + args.file_name_cfg.format(*formatter) + ".log"
        )
    )
    logger.info("Start running with config: \n{}".format(yaml.dump(cfg)))
    if args.gpu_preserve:
        logger.info("Preserving memory...")
        preserve_memory(args.preserve_percent)
        logger.info("Preserving memory done")
    
    # set seed
    seed = cfg["training"]["seed"]
    make_deterministic(seed)
    logger.info("Set seed : {}".format(seed))
    
    # get dataloaders
    logger.info("Loading datasets...")
    train_loader, val_loader, num_classes = get_dataloader(cfg)
    logger.info("num_classes: {}".format(num_classes))
    #set device loger writer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger_log = utils.logger.get_logger('distill_%s' % ('LTB_s2'))
    tb_writer =SummaryWriter(
        log_dir=os.path.join(
            args.logdir,
            "tf-logs",
            args.file_name_cfg.format(*formatter)
        ),
        flush_secs=1
    )

    # get models
    logger.info("Loading teacher {} and student {}...".format(
        cfg["kd"]["teacher"]["name"],cfg["kd"]["student"]["name"]))
    model_t = get_teacher(cfg, num_classes).to(device)
    model_t.eval()
    
    model_s = LEARNTOBRANCH_Deep(
        dataset=cfg["dataset"]["name"], 
        num_attributes=cfg["dataset"]["num_classes"],
        loss_method= cfg["model"]["loss_method"]).to(device)
    branch_params_list = list(map(id, model_s.branch_2.parameters())) + list(map(id, model_s.branch_3.parameters())) + \
                         list(map(id, model_s.branch_4.parameters()))
    global_params = filter(lambda p: id(p) not in branch_params_list, model_s.parameters())
   
    if args.stage == 's1':
        branch_params = filter(lambda p: id(p) in branch_params_list, model_s.parameters())
        params = [
            {"params": global_params, "lr": cfg["training"]["lr_global"]},
            {"params": branch_params, "lr": cfg["training"]["lr_branch"]},
        ]
    elif args.stage == 's2':
        params = global_params
    model_s.eval()
    logger.info(model_s)
    
    #get model dict
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
        tb_writer= tb_writer,
        device=device
    )
    trainable_dict.update(loss_trainable_dict)
    assert "teacher" not in trainable_dict.keys(), "teacher is not trainable"

    # set optimizer
    if args.stage == 's1':
        optimizer = torch.optim.Adam(params,
                            weight_decay=float(cfg["training"]["optimizer"]["weight_decay"]))
    elif args.stage == 's2':
        optimizer = torch.optim.SGD(
                params=global_params,
                lr=cfg["training"]["lr_stage2"],
                weight_decay=cfg["training"]["optimizer"]["weight_decay_stage2"],
                momentum=cfg["training"]["optimizer"]["momentum"])

        checkpoint = torch.load(cfg["model"]["pretrained"])
        # print(checkpoint.keys())
        model_s.load_state_dict(checkpoint)
        logger.info("=> loaded checkpoint '{}'".format(cfg["model"]["pretrained"]))
        model_s._initialize_weights()


    # set lr_scheduler 
    lr_scheduler = MultiStepLR(
        optimizer=optimizer,
        milestones=cfg["training"]["lr_decay_epochs"],
        gamma=cfg["training"]["lr_decay_rate"]
    )
    # append teacher after optimizer to avoid weight_decay
    module_dict["teacher"] = model_t.to(device)

    distiller = LTBDistiller(logger_log , tb_writer)
    
    distiller.setup(cfg = cfg, train_loader = train_loader,val_loader=val_loader, module_dict=module_dict,\
                    criterion_dict=criterion_dict, optimizer=optimizer,device=device,ckpt_dir = ckpt_dir,stage = args.stage)
    
    distiller.add_callback( 
        engine.DefaultEvents.AFTER_STEP(every=10), 
        callbacks=callbacks.MetricsLogging(keys=('total_loss', 'top_1', 'top_5','lr')))
    
    distiller.add_callback(
        engine.DefaultEvents.AFTER_STEP,
        callbacks=callbacks.LRSchedulerCallback(schedulers=[lr_scheduler]))
    
    distiller.run(start_iter=0, max_iter=len(train_loader)*(cfg["training"]["epochs"]+1))
    
if __name__ == "__main__":
    main()