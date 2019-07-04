import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.realpath(__file__)))))


from kamal.datasets import Cityscapes, CamVid
from kamal.recombination import combine_models, conv2d_combine_fn
from kamal.models.ae import RoundQuantizer, RangeQuantizer

from copy import deepcopy
from collections import namedtuple

_QuantizerOutput = namedtuple(
    '_QuantizerOutput', ['qbar', 'qsoft', 'qhard', 'symbols'])


class MultiTaskCamVid(CamVid):
    def __getitem__(self, idx):
        images, targets = super(MultiTaskCamVid, self).__getitem__(idx)
        return images, images, targets


class MultiTaskCityscapes(Cityscapes):
    def __getitem__(self, idx):
        images, targets = super(MultiTaskCityscapes, self).__getitem__(idx)
        return images, images, targets


class HybridNet(nn.Module):
    def __init__(self, encoder, decoders, code_chan):
        super(HybridNet, self).__init__()
        if not isinstance(decoders, nn.ModuleList):
            decoders = nn.ModuleList(decoders)

        self.encoder = encoder
        self.decoders = decoders

        bottle_neck = 16

        self.to_bottle_neck = nn.Sequential(
            nn.Conv2d(sum(code_chan), bottle_neck, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottle_neck),
            nn.LeakyReLU(),
            nn.Conv2d(bottle_neck, bottle_neck, kernel_size=1, bias=False)
        )

        adaptors = []
        for dec, chan in zip(decoders, code_chan):
            adaptors.append(
                nn.Sequential(
                    nn.Conv2d(bottle_neck, chan, kernel_size=1, bias=False),
                    nn.BatchNorm2d(chan),
                    nn.LeakyReLU(),
                    nn.Conv2d(chan, chan, kernel_size=1, bias=False)
                )
            )

        self.adaptors = nn.ModuleList(adaptors)

        self.quantizer = RangeQuantizer(6)

    def forward(self, x):
        x = x.repeat(1, len(self.decoders), 1, 1)
        x, _ = self.encoder(x)
        x = self.to_bottle_neck(x)

        q = self.quantizer(x)

        self.code = q
        output = []
        for adaptor, dec in zip(self.adaptors, self.decoders):
            dec_out = dec(adaptor(q.qbar))
            output.append(dec_out)
        return output
