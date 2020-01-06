import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import BasicBlock, DenseBlock, DiffClamp
from .quantizer import Quantizer

from collections import namedtuple

from torch.autograd import Function
import math

EncoderOutput = namedtuple(
    'EncoderOutput', ['qbar', 'qhard', 'symbols', 'z', 'heatmap'])


class _ResBlocks(nn.Module):
    def __init__(self, planes, num_block):
        super(_ResBlocks, self).__init__()
        body = []
        for i in range(num_block):
            body.append(BasicBlock(inplanes=planes, planes=planes))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x) + x


class AutoEncoder(nn.Module):
    def __init__(self, C=32, M=128, num_res_blocks=5, in_chan=3, out_chan=3, quantizer=None, sigmoid_out=False, heatmap=False, num_symbols=6, dropout_p=0.0):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(
            C=C, M=M, num_res_blocks=num_res_blocks, in_chan=in_chan, heatmap=heatmap)
        self.decoder = Decoder(C=C, M=M, num_res_blocks=num_res_blocks,
                               out_chan=out_chan, sigmoid_out=sigmoid_out, dropout_p=dropout_p)
        self.quantizer = quantizer
        # self.init_weight()

    def forward(self, x, **kargs):
        enc_out, heatmap = self.encoder(x)

        if self.quantizer is not None:
            code = self.quantizer(enc_out)
            code = EncoderOutput(code.qbar, code.qhard,
                                 code.symbols, enc_out, heatmap)
        else:
            code = EncoderOutput(enc_out, enc_out, None, enc_out, heatmap)

        self.code = code

        out = self.decoder(code.qbar)
        return out

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1e-2)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DiffCeil(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.ceil(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


class Encoder(nn.Module):
    """ Encoder
    """

    def __init__(self, C=32, M=128, num_res_blocks=5, in_chan=3, heatmap=False):
        self.heatmap = heatmap

        super(Encoder, self).__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=M//2,
                      kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(M//2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=M//2, out_channels=M,
                      kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(M),
            nn.LeakyReLU(inplace=True)
        )

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(_ResBlocks(M, num_block=3))
        res_blocks.append(BasicBlock(M, M))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.out_conv = nn.Conv2d(in_channels=M, out_channels=C+int(
            heatmap == True), kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.res_blocks(x)
        x = self.out_conv(x)

        if self.heatmap == True:
            heatmap = self._get_heatmap(x)
            x = self._mask_with_heatmap(x, heatmap)
        else:
            heatmap = None

        return x, heatmap

    def _mask_with_heatmap(self, x, heatmap):
        x_wo_heatmap = x[:, 1:, :, :]
        return heatmap * x_wo_heatmap

    def _get_heatmap(self, bottleneck):
        C = bottleneck.shape[1]-1
        heatmap = bottleneck[:, 0, :, :]  # NHW
        heatmap2D = torch.sigmoid(heatmap) * C
        c = torch.arange(C, dtype=torch.float, device=bottleneck.device).view(
            C, 1, 1)  # C, 1, 1
        heatmap = heatmap2D.unsqueeze(1)  # N1HW
        heatmap3D = DiffCeil.apply(torch.relu(
            torch.min(heatmap-c, other=heatmap.new_ones(1))))
        return heatmap3D


class Decoder(nn.Module):
    """ Decoder
    """

    def __init__(self, C=32, M=128, num_res_blocks=5, out_chan=3, sigmoid_out=False, dropout_p=0.0):
        super(Decoder, self).__init__()
        self.in_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=C, out_channels=M, kernel_size=3,
                               stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(M),
            nn.LeakyReLU()
        )
        self.sigmoid_out = sigmoid_out

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(_ResBlocks(M, num_block=3))
        res_blocks.append(BasicBlock(M, M))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.out_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=M, out_channels=M//2, kernel_size=5,
                               stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(M//2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=M//2, out_channels=M//4, kernel_size=5,
                               stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(M//4),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(in_channels=M//4, out_channels=out_chan,
                      kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, q):
        x = self.in_conv(q)
        x = self.res_blocks(x)
        x = self.out_conv(x)
        if self.sigmoid_out:
            return torch.sigmoid(x)
        else:
            return x
