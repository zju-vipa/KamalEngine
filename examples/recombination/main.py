import ext_transforms as et
from tqdm import tqdm
import random
import numpy as np
import argparse

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.realpath(__file__)))))


from hybrid_net import *
from kamal.losses import CriterionsCompose, SoftFocalLoss, MS_SSIM_Loss
from kamal.metrics import StreamCompMetrics, StreamSegMetrics
from kamal.models.ae import AutoEncoder, BasicBlock
from kamal.datasets import Cityscapes
from kamal.recombination import prune
from kamal.recombination import combine_models, bn_combine_fn, conv2d_combine_fn, Recombination
from kamal.core import LayerParser, Estimator
from kamal.utils import Visualizer

from torch.utils import data
import torch.nn.functional as F
import torch.nn as nn
import torch


def l1_choose(layer, num):
    weight = layer.weight.data.cpu().numpy()
    L1_norm = np.abs(weight).mean(axis=(1, 2, 3))  # CO
    idxs = np.argsort(-L1_norm)
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
    vis = Visualizer(port=opts.vis_port, env='prune_info')
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
        model_state = torch.load(ckpt)['model_state']
        model_state = {k: v for k, v in model_state.items()
                       if k in model.state_dict()}
        model.load_state_dict(model_state)
        print("Model restored from %s" % ckpt)

    restore_model(ta, ta_ckpt)
    restore_model(tb, tb_ckpt)

    combine_parser = LayerParser((nn.Conv2d, conv2d_combine_fn),
                                 (nn.BatchNorm2d, bn_combine_fn),
                                 match_fn=lambda layers, layer_type: isinstance(layers[0], layer_type))
    combined_encoder = combine_models([ta.encoder, tb.encoder], combine_parser)
    hybrid_net = HybridNet(combined_encoder, [deepcopy(
        ta.decoder), deepcopy(tb.decoder)], code_chan=[12, 32])

    hybrid_net = hybrid_net.to(device)
    ta = ta.to(device)
    tb = tb.to(device)
    # print(combined_encoder)

    train_transform = et.ExtCompose([
        #et.ExtResize(size=(512, 1024)),
        et.ExtRandomCrop(size=512, pad_if_needed=True),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
    ])

    val_transform = et.ExtCompose([
        #et.ExtResize(size=(512, 1024)),
        et.ExtToTensor(),
    ])
    # train_dst = MultiTaskCityscapes(root=opts.data_root,
    #                                split='train',
    #                                mode='fine',
    #                                transform=train_transform)
    # val_dst = MultiTaskCityscapes(root=opts.data_root,
    #                                split='val',
    #                                mode='fine',
    #                                transform=val_transform)
    train_dst = MultiTaskCamVid(root=opts.data_root,
                                split='train',
                                transform=train_transform)
    val_dst = MultiTaskCamVid(root=opts.data_root,
                              split='val',
                              transform=train_transform)
    test_dst = MultiTaskCamVid(root=opts.data_root,
                               split='test',
                               transform=val_transform)
    trainval_dst = torch.utils.data.ConcatDataset([train_dst, val_dst])

    train_loader = data.DataLoader(
        trainval_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2)
    test_loader = data.DataLoader(
        test_dst, batch_size=1, shuffle=opts.batch_size, num_workers=2)

    metrics_a = StreamCompMetrics(data_range=1.0)
    metrics_b = StreamSegMetrics(11)

    num_params = sum(p.numel()
                     for p in hybrid_net.parameters() if p.requires_grad)
    print("Params: %d" % num_params)

    for i in range(40):
        print("Iter %d" % i)

        prune_model(hybrid_net, i)

        num_params = sum(p.numel()
                         for p in hybrid_net.parameters() if p.requires_grad)
        print("Params: %d" % num_params)
        vis.vis_scalar('params', i, num_params)
        optimizer = torch.optim.Adam([{"params": hybrid_net.encoder.parameters()},
                                      {"params": hybrid_net.decoders.parameters()},
                                      {"params": hybrid_net.adaptors.parameters(),
                                       'lr': 1e-3},
                                      {"params": hybrid_net.to_bottle_neck.parameters(), 'lr': 1e-3}],
                                     lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.1)
        loss_d = MS_SSIM_Loss(data_range=1.0, size_average=True, channel=3)
        loss_s = SoftFocalLoss()
        criterions = CriterionsCompose([loss_d, loss_s], weights=[
                                       1., 1.], tags=['MS_SSIM Loss', 'FL Loss'])
        hybrid_net = Recombination(hybrid_net, [None, tb]).fit(train_loader,
                                                               val_loader=test_loader,
                                                               criterions=criterions,
                                                               metrics=[
                                                                   metrics_a, metrics_b],
                                                               enable_vis=True,
                                                               ckpt_name='iter%02d' % i,
                                                               total_epochs=30,
                                                               lr=1e-4,
                                                               gpu_id='1',
                                                               optimizer=optimizer,
                                                               scheduler=scheduler,
                                                               env='prune_iter%d' % i,
                                                               port=opts.vis_port)
