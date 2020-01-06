import torch
import torch.nn.functional as F
import torch.nn as nn

from pytorch_msssim import ssim, ms_ssim, MS_SSIM, SSIM
from .functional import *

class KDLoss(nn.Module):
    """ KD Loss Function
    """
    def __init__(self, T=1.0, alpha=1.0, use_kldiv=False):
        super(KDLoss, self).__init__()
        self.T = T
        self.alpha = alpha
        self.kdloss = kldiv if use_kldiv else soft_cross_entropy


    def forward(self, logits, targets, hard_targets=None):
        loss = self.kdloss(logits, targets, T=self.T)
        if hard_targets is not None and self.alpha != 0.0:
            loss += self.alpha*F.cross_entropy(logits, hard_targets)
        return loss

class CFLLoss(nn.Module):
    """ Common Feature Learning Loss
        CFL Loss = MMD + beta * MSE
    """
    def __init__(self, sigmas, normalized=True, beta=1.0):
        super(CFLLoss, self).__init__()
        self.sigmas = sigmas
        self.normalized = normalized
        self.beta = beta

    def forward(self, hs, hts, fts_, fts):
        mmd = mse = 0.0
        for ht_i in hts:
            mmd += mmd_loss(hs, ht_i, sigmas=self.sigmas, normalized=self.normalized)
        for i in range(len(fts_)):
            mse += F.mse_loss(fts_[i], fts[i])
        return mmd + self.beta*mse

class PSNR_Loss(nn.Module):
    def __init__(self, data_range=1.0, size_average=True):
        super(PSNR_Loss, self).__init__()
        self.data_range = data_range
        self.size_average = size_average

    def forward(self, img1, img2):
        return 100 - psnr(img1, img2, size_average=self.size_average, data_range=self.data_range)


class MS_SSIM_Loss(MS_SSIM):
    def forward(self, img1, img2):
        return 100*(1 - super(MS_SSIM_Loss, self).forward(img1, img2))

class ScaleInvariantLoss(nn.Module):
    """This criterion is used in depth prediction task.

    **Parameters:**
        - **la** (int, optional): Default value is 0.5. No need to change.
        - **ignore_index** (int, optional): Value to ignore.

    **Shape:**
        - **inputs**: $(N, H, W)$.
        - **targets**: $(N, H, W)$.
        - **output**: scalar.
    """
    def __init__(self, la=0.5, ignore_index=0):
        super(ScaleInvariantLoss, self).__init__()
        self.la = la
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        size = inputs.size()
        if len(size) == 3:
            inputs = inputs.view(size[0], -1)
            targets = targets.view(size[0], -1)

        inv_mask = targets.eq(self.ignore_index)
        nums = (1-inv_mask.float()).sum(1)
        log_d = torch.log(inputs) - torch.log(targets)
        log_d[inv_mask] = 0

        loss = torch.div(torch.pow(log_d, 2).sum(1), nums) - \
            self.la * torch.pow(torch.div(log_d.sum(1), nums), 2)

        return loss.mean()

class AngleLoss(nn.Module):
    """This criterion is used in surface normal prediction task.

    **Shape:**
        - **inputs**: $(N, 3, H, W)$. Predicted space vector for each pixel. Must be formalized before.
        - **targets**: $(N, 3, H, W)$. Ground truth. Must be formalized before.
        - **masks**: $(N, 1, H, W)$. One for valid pixels, else zero.
        - **output**: scalar.
    """
    def forward(self, inputs, targets, masks):
        nums = masks.sum(dim=[1,2,3])

        product = (inputs * targets).sum(1, keepdim=True)
        loss = -torch.div((product * masks.float()).sum([1,2,3]), nums)
        return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()
