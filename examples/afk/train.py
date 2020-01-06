import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from kamal.vision.datasets import voc
from kamal.vision.models import resnet, densenet
from kamal.metrics import stream_metrics
import sys
import os
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
    parser.add_argument("--data_root", type=str, default='/nfs/jyx/database/')
    parser.add_argument("--dataset", type=str, default='voc')
    parser.add_argument("--year", type=str, default='2007')
    parser.add_argument("--vis_env", type=str, default='Classification_2007_joint')
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--step_size", type=float, default=40 )
    parser.add_argument("--gamma", type=float, default=0.5 )
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--gpu_id", type=str, default='1')
    parser.add_argument("--epochs", type=int, default=90 )
    parser.add_argument("--init_ckpt", type=str, nargs = '*', default=['teacher_dense_07_1.pth', 'teacher_dense_07_2.pth', 
        "./checkpoints/afk/2007_joint/block[1, 1]_lr0.1_stepsize40_gamma0.5/82.pth"
    ])
    parser.add_argument("--ckpt", type=str, default='./checkpoints/afk/2007_joint')
    parser.add_argument("--phase", type=str, choices=['block', 'finetune'], default='block', 
                        help="""The 'block' phase is the step 2 in the paper, to train block by block. The layers of branches are fixed.
                                The 'finetune' phase is the step 4 in the paper. All layers are trainable.""")
    parser.add_argument("--indices", type=int, nargs='*', default=[2, 2], 
                        help="""Where to branch out for each task.""")
    parser.add_argument("--split_size", type=str, nargs = '*', default=[10, 10])
    parser.add_argument("--arange_indices", type=str, nargs = '*', default=[0,1,2,3,10,4,5,11,12,13,14,15,6,7,8,16,17,18,9,19])
    return parser

def train( cur_epoch, criterions, teacher_models, joint_model, split_size, optim, loader, device, vis, scheduler=None, print_interval=15):
    """Train and return epoch loss"""
    if scheduler is not None:
        scheduler.step()
    
    print("Epoch %d, lr = %f"%(cur_epoch, optim.param_groups[0]['lr']))
    epoch_loss = 0.0
    interval_loss = 0.0

    for cur_step, sample_batched in enumerate( loader ):
        images = sample_batched[0].to(device)
        labels = sample_batched[1].to(device)
        raw_labels = (labels == 1).float()
        optim.zero_grad()
        with torch.no_grad():
            teacher_outputs = [teacher_model(images) for teacher_model in teacher_models]
        joint_outputs = torch.split(joint_model(images), split_size, dim=1)

        losses = [criterions[i](joint_outputs[i], (torch.sigmoid(teacher_outputs[i])>0.5).float()) for i in range(len(teacher_models))]

        loss = reduce(lambda x,y:x+y, losses)
        loss.backward()
        optim.step()

        np_loss = loss.data.cpu().numpy()
        epoch_loss+=np_loss
        interval_loss+=np_loss
        pre_steps = cur_epoch * len(loader)
        if (cur_step + 1) % print_interval==0:
            interval_loss = interval_loss/print_interval
            print("Epoch {}, Batch {}/{}, Loss={}".format(cur_epoch, cur_step+1, len(loader), interval_loss))

            vis_images = sample_batched[0].numpy()
            vis_labels = sample_batched[1].numpy()
            vis_preds_list = [torch.sigmoid(output).data.cpu().numpy() for output in joint_outputs]
            vis_teachers_list = [torch.sigmoid(output).data.cpu().numpy() for output in teacher_outputs]

            vis.images(vis_images, nrow=3, win='images')
            vis.text(np.array2string(vis_labels, precision=2, separator='\t'), win='labels')
            for i, vis_preds in enumerate(vis_preds_list):
                vis.text(np.array2string(vis_preds, precision=2, separator='\t'), win='preds_{}'.format(i))    
            for i, vis_teachers in enumerate(vis_teachers_list):
                vis.text(np.array2string(vis_teachers, precision=2, separator='\t'), win='teachers_{}'.format(i))   
            for i, l in enumerate(losses):
                vis.line(X=[cur_step + pre_steps], Y=[l.data.cpu().numpy()], win='loss_{}'.format(i), update='append' if (cur_step + pre_steps) else None, opts=dict(title='loss_{}'.format(i)))
            vis.line(X=[cur_step + pre_steps], Y=[interval_loss], win='interval_loss', update='append' if (cur_step + pre_steps) else None, opts=dict(title='interval_loss'))
            vis.line(X=[cur_step + pre_steps], Y=[optim.param_groups[0]['lr']], win='learning_rate', update='append' if (cur_step + pre_steps) else None, opts=dict(title='learning_rate'))

            interval_loss=0.0

    return epoch_loss / len(loader)

def validate(joint_model, arange_indices, loader, device, metrics):
    metrics.reset()

    with torch.no_grad():
        for cur_step, sample_batched in tqdm(enumerate( loader ), mininterval=10):
            images = sample_batched[0].to(device)
            labels = sample_batched[1].to(device)
            weights = (labels != 0)
            labels[labels == -1] = 0

            # Recover label order.
            outputs = joint_model(images)
            outputs = torch.cat([outputs[:, idx:idx+1] for idx in arange_indices], dim=1)
            outputs = (torch.sigmoid(outputs) > 0.5).data.cpu().numpy()
            labels = labels.data.cpu().numpy()
            weights = weights.data.cpu().numpy()
            metrics.update(outputs, labels, weights)

    return metrics.get_results()

def main():
    opts = get_parser().parse_args()
    print(opts)

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vis = visdom.Visdom(env=opts.vis_env)
    
    ckpt_dir = os.path.join(opts.ckpt, '{}{}_lr{}_stepsize{}_gamma{}'.format(opts.phase, opts.indices, opts.lr, opts.step_size, opts.gamma))
    mkdir(ckpt_dir)

    if opts.dataset == 'voc':
        if opts.year == '2007':
            splits = ('trainval', 'test')
        elif opts.year == '2012':
            splits = ('train', 'val')

        train_ds = voc.VOCClassification(opts.data_root, year=opts.year, split=splits[0], transforms=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.6),
            transforms.Resize((400, 500)),
            transforms.ToTensor()
        ]), target_transforms=transforms.Compose([
            transforms.Lambda(lambda x: torch.Tensor(x)),
        ]))

        val_ds = voc.VOCClassification(opts.data_root, year=opts.year, split=splits[1], transforms=transforms.Compose([
            transforms.Resize((400, 500)),
            transforms.ToTensor(),
        ]), target_transforms=transforms.Compose([
            transforms.Lambda(lambda x: torch.Tensor(x)),
        ]))
            
    else:
        raise RuntimeError('Invalid dataset type.')

    train_loader = data.DataLoader(train_ds, batch_size=opts.batch_size, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(val_ds, batch_size=opts.batch_size, shuffle=False, num_workers=4)

    # teacher_model_1 = resnet.resnet50(num_classes=10)
    # teacher_model_2 = resnet.resnet50(num_classes=10)
    teacher_model_1 = densenet.DenseNet(block_config=(12,12,12), compression=1, num_init_features=16, bn_size=1, 
                    num_classes=10, small_inputs=False, efficient=True)
    teacher_model_2 = densenet.DenseNet(block_config=(12,12,12), compression=1, num_init_features=16, bn_size=1, 
                    num_classes=10, small_inputs=False, efficient=True)

    checkpoint = torch.load(opts.init_ckpt[0])
    teacher_model_1.load_state_dict(checkpoint["model_state"])
    print("Teacher model restored from %s"%opts.init_ckpt[0])
    del checkpoint

    checkpoint = torch.load(opts.init_ckpt[1])
    teacher_model_2.load_state_dict(checkpoint["model_state"])
    print("Teacher model restored from %s"%opts.init_ckpt[1])
    del checkpoint

    # joint_model = resnet.JointResNet([teacher_model_1, teacher_model_2], [3,4,6,3], opts.indices, phase=opts.phase)
    joint_model = densenet.JointDenseNet([teacher_model_1, teacher_model_2], opts.indices, phase=opts.phase)

    teacher_model_1.to(device)
    teacher_model_2.to(device)
    joint_model.to(device)

    metrics = stream_metrics.StreamClassificationMetrics()

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

    criterions = [nn.BCEWithLogitsLoss() for i in range(2)]

    # Restore
    cur_epoch = 0
    if len(opts.init_ckpt) > 2:
        checkpoint = torch.load(opts.init_ckpt[2])
        joint_model.load_state_dict(checkpoint["model_state"])
        if checkpoint["indices"] == opts.indices and checkpoint["phase"] == opts.phase:
            cur_epoch = checkpoint["epoch"]+1
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        print("Model restored from %s"%opts.init_ckpt[2])
        del checkpoint
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


    teacher_model_1.eval()
    teacher_model_2.eval()
    print(cur_epoch)
    while cur_epoch < opts.epochs:
        joint_model.train()
        epoch_loss = train(cur_epoch=cur_epoch, 
                            criterions=criterions, 
                            teacher_models=[teacher_model_1, teacher_model_2],
                            joint_model=joint_model, 
                            split_size=opts.split_size,
                            optim=optimizer, 
                            loader=train_loader, 
                            device=device, 
                            vis=vis,
                            scheduler=scheduler)
        print("End of Epoch %d/%d, Average Loss=%f"%(cur_epoch, opts.epochs, epoch_loss))

        save_ckpt(os.path.join(ckpt_dir, '{}.pth'.format(cur_epoch)))

        print("validate on val set...")
        joint_model.eval()
        val_scores = validate(joint_model=joint_model,
                                arange_indices=opts.arange_indices,
                                loader=val_loader, 
                                device=device, 
                                metrics=metrics,
                            )
        
        print(metrics.to_str(val_scores))

        cur_epoch+=1

if __name__ == '__main__':
    main()
