
import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_mmd(f1, f2, sigmas, normalized=False):
    if len(f1.shape) != 2:
        N, C, H, W = f1.shape
        f1 = f1.view(N, -1)
        N, C, H, W = f2.shape
        f2 = f2.view(N, -1)

    if normalized == True:
        f1 = F.normalize(f1, p=2, dim=1)
        f2 = F.normalize(f2, p=2, dim=1)

    return mmd_rbf2(f1, f2, sigmas=sigmas)


def mmd_rbf2(x, y, sigmas=None):
    N, _ = x.shape
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    K = L = P = 0.0
    XX2 = rx.t() + rx - 2*xx
    YY2 = ry.t() + ry - 2*yy
    XY2 = rx.t() + ry - 2*zz

    if sigmas is None:
        sigma2 = torch.mean((XX2.detach()+YY2.detach()+2*XY2.detach()) / 4)
        sigmas2 = [sigma2/4, sigma2/2, sigma2, sigma2*2, sigma2*4]
        alphas = [1.0 / (2 * sigma2) for sigma2 in sigmas2]
    else:
        alphas = [1.0 / (2 * sigma**2) for sigma in sigmas]

    for alpha in alphas:
        K += torch.exp(- alpha * (XX2.clamp(min=1e-12)))
        L += torch.exp(- alpha * (YY2.clamp(min=1e-12)))
        P += torch.exp(- alpha * (XY2.clamp(min=1e-12)))

    beta = (1./(N*(N)))
    gamma = (2./(N*N))

    return F.relu(beta * (torch.sum(K)+torch.sum(L)) - gamma * torch.sum(P))

