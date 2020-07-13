
import torch
import torch.nn.functional as F
import torch.nn as nn

def kldiv(logits, targets, T=1.0):
    """ Cross Entropy for soft targets
    
    Parameters:
        - logits (Tensor): logits score (e.g. outputs of fc layer)
        - targets (Tensor): logits of soft targets
        - T (float): temperature　of distill
        - reduction: reduction to the output
    """
    p_targets = F.softmax(targets/T, dim=1)
    logp_logits = F.log_softmax(logits/T, dim=1)
    kld = F.kl_div(logp_logits, p_targets, reduction='none') * (T**2)
    return kld.sum(1).mean()

def jsdiv(logits, targets, T=1.0, reduction='mean'):
    p = F.softmax(logits, dim=1)
    q = F.softmax(targets, dim=1)
    log_m = torch.log( (p+q) / 2 )
    return 0.5* ( F.kl_div( log_m,  p, reduction=reduction) + F.kl_div( log_m, q, reduction=reduction) )

def mmd_loss(f1, f2, sigmas, normalized=False):
    if len(f1.shape) != 2:
        N, C, H, W = f1.shape
        f1 = f1.view(N, -1)
        N, C, H, W = f2.shape
        f2 = f2.view(N, -1)

    if normalized == True:
        f1 = F.normalize(f1, p=2, dim=1)
        f2 = F.normalize(f2, p=2, dim=1)

    return _mmd_rbf2(f1, f2, sigmas=sigmas)

def psnr(img1, img2, size_average=True, data_range=255):
    N = img1.shape[0]
    mse = torch.mean(((img1-img2)**2).view(N, -1), dim=1)
    psnr = torch.clamp(torch.log10(data_range**2 / mse) * 10, 0.0, 99.99)
    if size_average == True:
        psnr = psnr.mean()
    return psnr

def soft_cross_entropy(logits, targets, T=1.0, size_average=True):
    """ Cross Entropy for soft targets
    
    **Parameters:**
        - **logits** (Tensor): logits score (e.g. outputs of fc layer)
        - **targets** (Tensor): logits of soft targets
        - **T** (float): temperature　of distill
        - **size_average**: average the outputs
    """
    p_targets = F.softmax(targets/T, dim=1)
    logp_pred = F.log_softmax(logits/T, dim=1)
    ce = torch.sum(-p_targets * logp_pred, dim=1)
    if size_average:
        return ce.mean() * T * T
    else:
        return ce * T * T

def _mmd_rbf2(x, y, sigmas=None):
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