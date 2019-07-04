import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from kamal.layer_wise_ka import *
from kamal.models import *
from kamal.datasets import *
from kamal.metrics import StreamSegMetrics, StreamDepthMetrics, StreamAngleMetrics
from kamal.losses import ScaleInvariantLoss, AngleLoss
from joint_net import JointNet
from torchvision import transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import argparse
import random
import numpy as np
from tqdm import tqdm
import visdom
from functools import reduce


def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='../database/')
    parser.add_argument("--dataset", type=str, default='nyu')
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--step_size", type=float, default=70 )
    parser.add_argument("--gamma", type=float, default=0.3 )
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--random_seed", type=int, default=1337)
    parser.add_argument("--epochs", type=int, default=80 )
    parser.add_argument("--init_ckpt", type=str, nargs = '*')
    parser.add_argument("--ckpt", type=str, default='./checkpoints/joint/')
    parser.add_argument("--phase", type=str, choices=['block', 'finetune'], default='block',
                        help="""The 'block' phase is the step 2 in the paper, to train block by block. The layers of branches are fixed.
                            The 'finetune' phase is the step 4 in the paper. All layers are trainable.
                        """)
    parser.add_argument("--indices", type=int, nargs='*', default=[3, 1],
                            help="""Where to branch out for each task.""")
    return parser

def train( cur_epoch, criterions, teacher_models, joint_model, split_size, optim, train_loader, device, vis, scheduler=None, print_interval=10):
    """Train and return epoch loss"""
    if scheduler is not None:
        scheduler.step()
    
    print("Epoch %d, lr = %f"%(cur_epoch, optim.param_groups[0]['lr']))
    epoch_loss = 0.0
    interval_loss = 0.0

    for cur_step, sample_batched in enumerate( train_loader ):
        images = sample_batched[0].to(device, dtype=torch.float32)
        labels = sample_batched[1].to(device, dtype=torch.long)
        depths = sample_batched[2].to(device)
        normals = sample_batched[3].to(device, dtype=torch.float32)
        masks = sample_batched[4].to(device, dtype=torch.float32)

        # N, C, H, W
        optim.zero_grad()

        with torch.no_grad():
            teacher_outputs = [teacher_model(images) for teacher_model in teacher_models]

        joint_outputs = torch.split(joint_model(images), split_size, dim=1)

        losses = []
        teacher_outputs[0] = torch.max(teacher_outputs[0], 1)[1]
        losses.append(criterions[0](F.log_softmax(joint_outputs[0], dim=1), teacher_outputs[0]))

        mask_size = joint_outputs[1].shape
        teacher_outputs[1] = F.normalize(teacher_outputs[1], dim=1)
        losses.append(criterions[1](F.normalize(joint_outputs[1], dim=1), 
                                    teacher_outputs[1],
                                    mask=torch.ones([mask_size[0], 1, mask_size[2], mask_size[3]]).to(device)
                                    ))

        loss = losses[0] + losses[1]
        
        loss.backward()
        optim.step()

        np_loss = loss.data.cpu().numpy()
        epoch_loss+=np_loss
        interval_loss+=np_loss
        pre_steps = cur_epoch * len(train_loader)
        if (cur_step + 1) % print_interval==0:
            interval_loss = interval_loss/print_interval
            print("Epoch {}, Batch {}/{}, Loss={}".format(cur_epoch, cur_step+1, len(train_loader), interval_loss))

            vis_images = sample_batched[0].numpy()
            vis_labels = sample_batched[1].numpy().astype(np.uint8)
            vis_labels = np.expand_dims(vis_labels, axis=1) * 19
            vis_normals = (sample_batched[3].numpy() + 1) / 2.
            vis_masks = sample_batched[4].numpy()

            vis_teacher_1 = teacher_outputs[0].cpu().data.numpy()
            vis_teacher_1 = np.expand_dims(vis_teacher_1, axis=1) * 19
            vis_teacher_2 = ((teacher_outputs[1] + 1) / 2).data.cpu().numpy()

            vis_preds_1 = torch.max(joint_outputs[0], 1)[1].cpu().data.numpy()
            vis_preds_1 = np.expand_dims(vis_preds_1, axis=1) * 19
            vis_preds_2 = ((F.normalize(joint_outputs[1], dim=1) + 1) / 2).data.cpu().numpy()

            vis.images(vis_images, nrow=3, win='images')
            vis.images(vis_labels, nrow=3, win='labels')
            vis.images(vis_normals, nrow=3, win='normals')
            vis.images(vis_masks, nrow=3, win='masks')
            vis.images(vis_teacher_1, nrow=3, win='seg_teacher')
            vis.images(vis_teacher_2, nrow=3, win='normal_teacher')
            vis.images(vis_preds_1, nrow=3, win='seg_predictions')
            vis.images(vis_preds_2, nrow=3, win='normal_predictions')
            vis.line(X=[cur_step + pre_steps], Y=[interval_loss], win='interval_loss', update='append' if (cur_step + pre_steps) else None, opts=dict(title='interval_loss'))
            for i, l in enumerate(losses):
                vis.line(X=[cur_step + pre_steps], Y=[l.data.cpu().numpy()], win='loss_{}'.format(i), update='append' if (cur_step + pre_steps) else None, opts=dict(title='loss_{}'.format(i)))
            vis.line(X=[cur_step + pre_steps], Y=[optim.param_groups[0]['lr']], win='learning_rate', update='append' if (cur_step + pre_steps) else None, opts=dict(title='learning_rate'))

            interval_loss=0.0

    return epoch_loss / len(train_loader)

def validate(joint_model, split_size, loader, device, metrics):
    for m in metrics:
        m.reset()

    with torch.no_grad():
        for cur_step, sample_batched in tqdm(enumerate( loader )):
            images = sample_batched[0].to(device, dtype=torch.float32)
            labels = sample_batched[1].to(device, dtype=torch.long)
            # depths = sample_batched[2].to(device)
            normals = sample_batched[3].to(device, dtype=torch.float32)
            masks = sample_batched[4].to(device, dtype=torch.float32)

            joint_outputs = torch.split(joint_model(images), split_size, dim=1)

            preds1 = joint_outputs[0].data.max(dim=1)[1].cpu().numpy()
            targets1 = labels.data.cpu().numpy()
            metrics[0].update(preds1, targets1)

            preds2 = F.normalize(joint_outputs[1], dim=1)
            preds2 = preds2.data.cpu().numpy()
            targets2 = F.normalize(normals, dim=1)
            targets2 = targets2.data.cpu().numpy()
            masks = masks.data.cpu().numpy()
            metrics[1].update(preds2, targets2, masks)

    return [m.get_results() for m in metrics]

def main():
    opts = get_parser().parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vis = visdom.Visdom(env='joint_{}_{}'.format(opts.phase, opts.indices))
    
    set_seed(opts.random_seed)

    ckpt_dir = os.path.join(opts.ckpt, '{}_{}_lr{}_stepsize{}_gamma{}'.format(opts.phase, opts.indices, opts.lr, opts.step_size, opts.gamma))
    mkdir(ckpt_dir)

    if opts.dataset == 'nyu':
        num_classes = 13
        train_ds = NYUv2(os.path.join(opts.data_root, 'NYU'), 'train', num_classes,
                    transforms=transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.ColorJitter(0.7),
                        transforms.ToTensor()
                    ]),
                    target_transforms=[
                        transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.Lambda(lambda target: torch.from_numpy(np.array(target)))
                        ]),
                        transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.Lambda(lambda depth: torch.from_numpy(np.array(depth)).float() / 1000.)
                        ]),
                        transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Lambda(lambda normal: normal * 2 - 1)
                        ]),
                        transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                        ]),
                    ], ds_type='labeled')
        val_ds = NYUv2(os.path.join(opts.data_root, 'NYU'), 'test', num_classes, 
                    transforms=transforms.Compose([
                        transforms.ToTensor()
                    ]),
                    target_transforms=[
                        transforms.Lambda(lambda target: torch.from_numpy(np.array(target))),
                        transforms.Lambda(lambda depth: torch.from_numpy(np.array(depth)).float() / 1000.),
                        transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Lambda(lambda normal: normal * 2 - 1)
                        ]),
                        transforms.ToTensor()
                    ], ds_type='labeled')
    else:
        pass

    train_loader = data.DataLoader(train_ds, batch_size=opts.batch_size, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(val_ds, batch_size=opts.batch_size, shuffle=False, num_workers=4)

    teacher_seg_model = SegNet(n_classes=num_classes, positive=True)
    teacher_normal_model = SegNet(n_classes=3, positive=False)

    print('phase: {}, indices: {}'.format(opts.phase, opts.indices))

    joint_model = JointNet(
        [teacher_seg_model, teacher_normal_model],
        indices=opts.indices,
        phase=opts.phase
    )

    teacher_seg_model = teacher_seg_model.to(device)
    teacher_normal_model = teacher_normal_model.to(device)
    joint_model = joint_model.to(device)

    metrics = [StreamSegMetrics(num_classes), StreamAngleMetrics(thresholds=[11.25, 22.5, 30])]

    params_1x = []
    params_10x = []
    for name, param in joint_model.named_parameters():
        if param.requires_grad:
            if 'fc' in name:
                params_10x.append(param)
            else:
                params_1x.append(param)

    optimizer = torch.optim.SGD(params=[{'params': params_1x,  'lr': opts.lr  },
                                        {'params': params_10x, 'lr': opts.lr*10 }],
                                   lr=opts.lr, weight_decay=1e-5, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=opts.gamma)
    criterions = [torch.nn.NLLLoss2d(ignore_index=255), AngleLoss()]

    # Restore
    cur_epoch = 0
    branch_checkpoint = {}
    if opts.init_ckpt is not None:
        checkpoint = torch.load(opts.init_ckpt[0])
        teacher_seg_model.load_state_dict(checkpoint["model_state"])
        for k in checkpoint['model_state']:
            if k[0:2] == 'up':
                joint_name = 'student_b_decoders_list.0.{}.{}'.format(5-int(k[2]), k[4:])
                branch_checkpoint[joint_name] = checkpoint['model_state'][k]

        print("Model restored from %s"%opts.init_ckpt[0])
        del checkpoint

        checkpoint = torch.load(opts.init_ckpt[1])
        teacher_normal_model.load_state_dict(checkpoint["model_state"])
        for k in checkpoint['model_state']:
            if k[0:2] == 'up':
                joint_name = 'student_b_decoders_list.1.{}.{}'.format(5-int(k[2]), k[4:])
                branch_checkpoint[joint_name] = checkpoint['model_state'][k]

        print("Model restored from %s"%opts.init_ckpt[1])
        del checkpoint

        if len(opts.init_ckpt) > 2:
            checkpoint = torch.load(opts.init_ckpt[2])
            joint_model.load_state_dict(checkpoint["model_state"])
            if checkpoint["indices"] == opts.indices and checkpoint["phase"] == opts.phase:
                cur_epoch = checkpoint["epoch"]+1
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                scheduler.load_state_dict(checkpoint["scheduler_state"])
            print("Joint model restored from %s"%opts.init_ckpt[2])
            del checkpoint
        else:
            # Recover parameters of branches.
            # Another way is to recover that of teacher models before constructing joint student model, 
            # which is much more convenient, and shown in 'train_offline.py'.
            joint_model.load_state_dict(branch_checkpoint, strict=False)
            print("Joint model restored from teachers: {}, {}".format(opts.init_ckpt[0], opts.init_ckpt[1]))

    else:
        print("[!] No Restoration")

    def save_ckpt(path):
        """ save current model
        """
        state = {
                    "epoch": cur_epoch,
                    "model_state": joint_model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "indices": opts.indices,
                    "phase": opts.phase,
        }
        torch.save(state, path)
        print( "Model saved as %s"%path )


    teacher_seg_model.eval()
    teacher_normal_model.eval()

    while cur_epoch < opts.epochs:
        joint_model.train()
        epoch_loss = train(cur_epoch=cur_epoch, 
                            criterions=criterions, 
                            joint_model=joint_model, 
                            teacher_models=[teacher_seg_model, teacher_normal_model],
                            split_size=[num_classes, 3],
                            optim=optimizer, 
                            train_loader=train_loader, 
                            device=device, 
                            vis=vis,
                            scheduler=scheduler)
        print("End of Epoch %d/%d, Average Loss=%f"%(cur_epoch, opts.epochs, epoch_loss))

        save_ckpt(os.path.join(ckpt_dir, '{}.pth'.format(cur_epoch)))

        print("validate on val set...")
        joint_model.eval()
        val_scores = validate(joint_model=joint_model,
                                split_size=[num_classes, 3],
                                loader=val_loader, 
                                device=device, 
                                metrics=metrics,
                            )
        
        for m, v in zip(metrics, val_scores):
            print(m.to_str(v))

        cur_epoch+=1

if __name__ == '__main__':
    main()
