
import torch
import torch.nn.functional as F
from .mmd import mmd_rbf2
import torch

def kldiv(logits, targets, T=1.0, reduction='mean'):
    """ Cross Entropy for soft targets
    
    Parameters:
        - logits (Tensor): logits score (e.g. outputs of fc layer)
        - targets (Tensor): logits of soft targets
        - T (float): temperature　of distill
        - reduction: reduction to the output
    """
    p_targets = F.softmax(targets/T, dim=1)
    logp_logits = F.log_softmax(logits/T, dim=1)
    return F.kl_div(logp_logits, p_targets, reduction=reduction)*T*T

def mmd_loss(f1, f2, sigmas, normalized=False):
    if len(f1.shape) != 2:
        N, C, H, W = f1.shape
        f1 = f1.view(N, -1)
        N, C, H, W = f2.shape
        f2 = f2.view(N, -1)

    if normalized == True:
        f1 = F.normalize(f1, p=2, dim=1)
        f2 = F.normalize(f2, p=2, dim=1)

    return mmd_rbf2(f1, f2, sigmas=sigmas)

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