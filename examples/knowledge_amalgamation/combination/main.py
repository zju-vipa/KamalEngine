from tqdm import tqdm
import random
import numpy as np
import argparse

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.realpath(__file__)))))

from hybrid import *
from kamal.criterion import KDLoss, MS_SSIM_Loss
from kamal.metrics import ReconstructionMetrics, SegmentationMetrics

from kamal.vision.models.ae import AutoEncoder, BasicBlock
from kamal.vision.datasets import CamVid

from kamal.amalgamation.combination import prune, CombinedModel
from kamal.utils import VisdomPlotter

from kamal.vision import seg_transforms as Tseg

from torch.utils import data
import torch.nn.functional as F
import torch.nn as nn
import torch

def l1_choose(layer, num):
    weight = layer.weight.data.cpu().numpy()
    L1_norm = np.abs(weight).mean(axis=(1, 2, 3))  # CO
    idxs = np.argsort(L1_norm)
    return idxs[:num]

def prune_model(model, cur_iter):
    encoder = model.encoder
    idxs = l1_choose(encoder.in_conv[0], 2)
    prune.prune_conv_layer(encoder.in_conv[0], idxs)
    prune.prune_bn_layer(encoder.in_conv[1], idxs)
    prune.prune_related_conv_layer(encoder.in_conv[3], idxs)
    # in_conv 3 bn 4 leakyrelu 5

    idxs = l1_choose(encoder.in_conv[3], 4)
    prune.prune_conv_layer(encoder.in_conv[3], idxs)
    prune.prune_bn_layer(encoder.in_conv[4], idxs)

    for m in encoder.res_blocks.modules():
        if isinstance(m, BasicBlock):
            prune.prune_related_conv_layer(m.conv1, idxs)
            #prune.prune_bn_layer(m.bn1, idxs)
            prune.prune_conv_layer(m.conv2, idxs)
            prune.prune_bn_layer(m.bn2, idxs)

            # prune blocks
            res_idxs = l1_choose(m.conv1, 4)
            prune.prune_conv_layer(m.conv1, res_idxs)
            prune.prune_bn_layer(m.bn1, res_idxs)
            prune.prune_related_conv_layer(m.conv2, res_idxs)

    prune.prune_related_conv_layer(encoder.out_conv, idxs)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def mkdir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)

def eval(hyber_net, test_loader, metric_comp, metric_seg, device=None ):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hyber_net.to(device).eval()

    metric_comp.reset()
    metric_seg.reset()

    with torch.no_grad():
        for i, (img, target) in enumerate(test_loader):
            img, target = img.to(device), target.to(device)
            comp_out, seg_out = hybrid_net( img )
            metric_comp.update( comp_out, img )
            metric_seg.update( seg_out.max(1)[1], target )

    return metric_comp.get_results(return_key_metric=True), metric_seg(return_key_metric=True)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str,
                        choices=['prune', 'finetune'], default='finetune')

    parser.add_argument("--data_root", type=str, default='./data/cityscapes')
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--random_seed", type=int, default=1337)
    parser.add_argument("--download", action='store_true', default=False)
    parser.add_argument("--only_kd", action='store_true', default=False)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--vis_port", type=str, default='13571')
    return parser


if __name__ == '__main__':
    opts = get_parser().parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vp = VisdomPlotter(port=opts.vis_port, env='prune_info')
    # Set up random seed

    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    ckpt_dir = 'checkpoints'
    ta_ckpt = os.path.join('pretrained', 'best_comp_imagenet_ae_12.pkl')
    tb_ckpt = os.path.join('pretrained', 'best_seg_camvid_ae_32.pkl')

    ta = AutoEncoder(num_res_blocks=2, M=128, C=12, out_chan=3)  # Comp
    tb = AutoEncoder(num_res_blocks=2, M=128, C=32, out_chan=11)  # Seg

    def restore_model(model, ckpt):
        model_state = torch.load(ckpt)
        model_state = {k: v for k, v in model_state.items() if k in model.state_dict()}
        model.load_state_dict(model_state)
        print("Model restored from %s" % ckpt)
    restore_model(ta, ta_ckpt)
    restore_model(tb, tb_ckpt)

    combined_encoder = CombinedModel([ta.encoder, tb.encoder])
    hybrid_net = HybridNet(combined_encoder, [deepcopy(ta.decoder), deepcopy(tb.decoder)], code_chan=[12, 32])
    hybrid_net = hybrid_net.to(device)
    ta = ta.to(device)
    tb = tb.to(device)

    train_transform = Tseg.Compose([
        #et.ExtResize(size=(512, 1024)),
        Tseg.RandomCrop(size=512, pad_if_needed=True),
        Tseg.RandomHorizontalFlip(),
        Tseg.ToTensor(),
    ])

    val_transform = Tseg.Compose([
        #et.ExtResize(size=(512, 1024)),
        Tseg.ToTensor(),
    ])

    train_dst = CamVid(root=opts.data_root,
                       split='train',
                       transforms=train_transform)
    val_dst = CamVid(root=opts.data_root,
                            split='val',
                            transform=train_transform)
    test_dst = CamVid(root=opts.data_root,
                            split='test',
                            transform=val_transform)
    trainval_dst = torch.utils.data.ConcatDataset([train_dst, val_dst])
    train_loader = data.DataLoader(trainval_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2)
    test_loader = data.DataLoader(test_dst, batch_size=1, shuffle=opts.batch_size, num_workers=2)

    metrics_a = ReconstructionMetrics(data_range=1.0)
    metrics_b = SegmentationMetrics(11)

    num_params = sum(p.numel() for p in hybrid_net.parameters() if p.requires_grad)
    print("Params: %d" % num_params)

    pruning_iters = 40
    for i in range(40):
        old_num_params = sum(p.numel() for p in hybrid_net.parameters() if p.requires_grad)
        print("Iter %d" % i)
        prune_model(hybrid_net, i)
        new_num_params = sum(p.numel() for p in hybrid_net.parameters() if p.requires_grad)
        print("Params: %d => %d" % (old_num_params, new_num_params))

        vp.vis_scalar('params', i, new_num_params)

        optimizer = torch.optim.Adam([{"params": hybrid_net.encoder.parameters()},
                                      {"params": hybrid_net.decoders.parameters()},
                                      {"params": hybrid_net.adaptors.parameters(), 'lr': 1e-3},
                                      {"params": hybrid_net.to_bottle_neck.parameters(), 'lr': 1e-3}],
                                      lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.1)
        criterion_comp = MS_SSIM_Loss(data_range=1.0, size_average=True, channel=3)
        criterion_seg = KDLoss()

        tb.eval()
        hybrid_net.train()

        total_epochs = 30
        for epoch in range( total_epochs ):
            for b, (img, target) in enumerate( train_loader ):
                img, target = img.to(device), target.to(device)
                optimizer.zero_grad()

                comp_out, seg_out = hybrid_net( img )
                with torch.no_grad:
                    soft_target = tb( img )
                loss_comp = criterion_comp( comp_out, img )
                loss_seg = criterion_seg( seg_out, soft_target, hard_targets=target)

                loss = loss_comp + loss_seg

                loss.backward()
                optimizer.step()
                if b % 10==0:
                    print("Pruning Iter %d/%d, Epoch %d/%d, Batch %d/%d, Loss=%.4f (comp=%.4f, seg=%.4f)"%( i, pruning_iters, epoch, total_epochs, b, len(train_loader), 
                                                                                                            loss.item(), loss_comp.item(), loss_seg.item() ))
            comp_score, seg_score = eval( hybrid_net, test_loader, metric_comp, metric_seg, device=device )
            print("[TEST] %s=%.4f, %s=%.4f"%( *comp_score, *seg_score ))
