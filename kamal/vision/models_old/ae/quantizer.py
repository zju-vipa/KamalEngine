import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

_QuantizerOutput = namedtuple(
    '_QuantizerOutput', ['qbar', 'qsoft', 'qhard', 'symbols'])


class RangeQuantizer(nn.Module):
    def __init__(self, num_symbols=6):
        super(RangeQuantizer, self).__init__()
        self.num_symbols = num_symbols

    def forward(self, x):
        qsoft = (self.num_symbols-1) * torch.sigmoid(x)
        qhard = torch.round(qsoft).detach()
        qbar = (qhard - qsoft).detach() + qsoft
        return _QuantizerOutput(qbar, qsoft, qhard, qhard.long())


class RoundQuantizer(nn.Module):
    def __init__(self):
        super(RoundQuantizer, self).__init__()

    def forward(self, x):
        qhard = torch.round(x).detach()
        qsoft = x
        qbar = (qhard - x).detach() + x
        return _QuantizerOutput(qbar, qsoft, qhard, qhard)


class DynamicQuantizer(nn.Module):
    def __init__(self, M=128, num_symbols=6, sigma=1.0):
        super(DynamicQuantizer, self).__init__()
        self.sigma = sigma
        self.num_symbols = num_symbols

        self.center_conv = nn.Sequential(
            nn.Conv2d(M, M//2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(M//2),
            nn.ReLU(),
            nn.Conv2d(M//2, M//4, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(M//4),
            nn.ReLU(),
            nn.Conv2d(M//4, num_symbols, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_symbols)
        )

    def forward(self, x, center_features):
        c = self.center_conv(center_features)
        centers = c.view(c.shape[0], c.shape[1], -1).mean(2)  # n x centers
        #centers = centers + torch.arange( self.num_symbols, dtype=x.dtype, device=x.device )

        #centers = (torch.round(centers) - centers).detach() + centers
        qsoft, qhard, symbols = self._quantize(x, centers)
        qbar = qsoft + (qhard - qsoft).detach()
        return _QuantizerOutput(qbar, qsoft, qhard, symbols), centers

    def _quantize(self, x, centers):
        sigma = self.sigma

        num_centers = centers.shape[-1]
        B, C, H, W = x.shape
        x = x.view(B, C, H*W, 1)
        _centers = centers.view(B, 1, 1, num_centers)

        #print(centers.view(B,1,1,num_centers).shape, x.shape)
        dist = (x-_centers)**2  # n, c, hw, num_centers

        phi_hard = F.softmax(-1e7 * dist, dim=-1)
        symbols_hard = torch.max(phi_hard, dim=-1, keepdim=True)[1]

        hardout = torch.stack([centers[i][symbols_hard[i]]
                               for i in range(B)], dim=0)

        phi_soft = F.softmax(-sigma * dist, dim=-1)
        softout = (phi_soft * _centers).sum(dim=3, keepdim=True)
        return tuple(map(lambda t: t.view(B, C, H, W), (softout, hardout, symbols_hard)))


class Quantizer(nn.Module):
    """ Quantizer
    """

    def __init__(self, num_centers=6, sigma=1.0):
        super(Quantizer, self).__init__()
        self.centers = nn.Parameter(data=torch.rand(
            num_centers)*4-2, requires_grad=True)
        self.sigma = sigma

    def forward(self, x):
        qsoft, qhard, symbols = self._quantize(x)
        qbar = qsoft + (qhard - qsoft).detach()
        return _QuantizerOutput(qbar, qsoft, qhard, symbols)

    def _quantize(self, x):
        centers = self.centers
        sigma = self.sigma

        num_centers = centers.shape[-1]
        B, C, H, W = x.shape
        x = x.view(B, C, H*W, 1)

        dist = (x-centers)**2

        phi_hard = F.softmax(-1e7 * dist, dim=-1)
        symbols_hard = torch.max(phi_hard, dim=-1, keepdim=True)[1]
        hardout = centers[symbols_hard]

        phi_soft = F.softmax(-sigma * dist, dim=-1)
        softout = (phi_soft * centers).sum(dim=3, keepdim=True)

        return tuple(map(lambda x: x.view(B, C, H, W), (softout, hardout, symbols_hard)))

    def from_symbols(self, symbols):
        return self.centers[symbols]
