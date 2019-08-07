import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from kamal.datasets import voc
from kamal.models import resnet, densenet
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


def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='/nfs/jyx/database/')
    parser.add_argument("--dataset", type=str, default='voc')
    parser.add_argument("--year", type=str, default='2007')
    # parser.add_argument("--classes", type=str, nargs='*', default=['aeroplane', 'bicycle', 'bird', 'boat', 'bus', 'car', 'horse', 'motorbike', 'person', 'train'])
    parser.add_argument("--classes", type=str, nargs='*', default=['bottle', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'pottedplant', 'sheep', 'sofa', 'tvmonitor'])
    # parser.add_argument("--vis_env", type=str, default='Classification_2007teacher_1')
    parser.add_argument("--vis_env", type=str, default='Classification_2007teacher_2')
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--step_size", type=float, default=30 )
    parser.add_argument("--gamma", type=float, default=0.2 )
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--random_seed", type=int, default=1337)
    parser.add_argument("--epochs", type=int, default=70 )
    parser.add_argument("--init_ckpt", type=str, default=None)
    # parser.add_argument("--ckpt", type=str, default='./checkpoints/afk/2007_teacher_1')
    parser.add_argument("--ckpt", type=str, default='./checkpoints/afk/2007_teacher_2')

    return parser

def train( cur_epoch, criterion, model, optim, loader, device, vis, scheduler=None, print_interval=30):
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

        # N, C, H, W
        optim.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, raw_labels)
        
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
            vis_preds = torch.sigmoid(outputs).data.cpu().numpy()

            vis.images(vis_images, nrow=3, win='images')
            vis.text(np.array2string(vis_labels, precision=2, separator='\t'), win='labels')
            vis.text(np.array2string(vis_preds, precision=2, separator='\t'), win='preds')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
            vis.line(X=[cur_step + pre_steps], Y=[interval_loss], win='interval_loss', update='append' if (cur_step + pre_steps) else None, opts=dict(title='interval_loss'))
            vis.line(X=[cur_step + pre_steps], Y=[optim.param_groups[0]['lr']], win='learning_rate', update='append' if (cur_step + pre_steps) else None, opts=dict(title='learning_rate'))

            interval_loss=0.0

    return epoch_loss / len(loader)

def validate(model, loader, device, metrics):
    metrics.reset()

    with torch.no_grad():
        for cur_step, sample_batched in tqdm(enumerate( loader ), mininterval=10):
            images = sample_batched[0].to(device)
            labels = sample_batched[1].to(device)
            weights = (labels != 0)
            labels[labels == -1] = 0

            outputs = model(images)
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
    
    ckpt_dir = os.path.join(opts.ckpt, 'lr{}_stepsize{}_gamma{}'.format(opts.lr, opts.step_size, opts.gamma))
    mkdir(ckpt_dir)
    indices = [voc.CLASSES.index(clas) for clas in opts.classes]

    def convert_clas(inputs):
        outputs = torch.cat([inputs[idx].unsqueeze_(-1) for idx in indices])
        return outputs

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
            transforms.Lambda(convert_clas),
        ]))

        val_ds = voc.VOCClassification(opts.data_root, year=opts.year, split=splits[1], transforms=transforms.Compose([
            transforms.Resize((400, 500)),
            transforms.ToTensor(),
        ]), target_transforms=transforms.Compose([
            transforms.Lambda(lambda x: torch.Tensor(x)),
            transforms.Lambda(convert_clas),
        ]))
            
    else:
        raise RuntimeError('Invalid dataset type.')

    train_loader = data.DataLoader(train_ds, batch_size=opts.batch_size, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(val_ds, batch_size=opts.batch_size, shuffle=False, num_workers=4)

    # model = resnet.resnet50(num_classes=len(opts.classes))
    model = densenet.DenseNet(block_config=(12,12,12), compression=1, num_init_features=16, bn_size=1, 
                    num_classes=len(opts.classes), small_inputs=False, efficient=True)
    
    model = model.to(device)

    metrics = stream_metrics.StreamClassificationMetrics()

    params_1x = []
    params_10x = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'fc' in name:
                params_10x.append(param)
            else:
                params_1x.append(param)

    optimizer = torch.optim.SGD(params=[{'params': params_1x,  'lr': opts.lr  },
                                        {'params': params_10x, 'lr': opts.lr*10 }],
                                   lr=opts.lr, weight_decay=1e-5, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=opts.gamma)

    criterion = nn.BCEWithLogitsLoss()

    # Restore
    cur_epoch = 0
    if opts.init_ckpt is not None:
        checkpoint = torch.load(opts.init_ckpt)
        model.load_state_dict(checkpoint["model_state"])
        cur_epoch = checkpoint["epoch"]+1
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        print("Model restored from %s"%opts.init_ckpt)
        del checkpoint
    else:
        print("[!] No Restoration")

    def save_ckpt(path):
        """ save current model
        """
        state = {
                    "epoch": cur_epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
        }
        torch.save(state, path)
        print( "Model saved as %s"%path )


    while cur_epoch < opts.epochs:
        model.train()
        epoch_loss = train(cur_epoch=cur_epoch, 
                            criterion=criterion, 
                            model=model, 
                            optim=optimizer, 
                            loader=train_loader, 
                            device=device, 
                            vis=vis,
                            scheduler=scheduler)
        print("End of Epoch %d/%d, Average Loss=%f"%(cur_epoch, opts.epochs, epoch_loss))

        save_ckpt(os.path.join(ckpt_dir, '{}.pth'.format(cur_epoch)))

        print("validate on val set...")
        model.eval()
        val_scores = validate(model=model,
                                loader=val_loader, 
                                device=device, 
                                metrics=metrics,
                            )
        
        print(metrics.to_str(val_scores))

        cur_epoch+=1

if __name__ == '__main__':
    main()
