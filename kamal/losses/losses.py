import torch
import torch.nn.functional as F
import torch.nn as nn

from ..misc.ssim import MS_SSIM
from ..misc.psnr import psnr
from .functional import *

from .mmd import calc_mmd


class CriterionsCompose(object):
    """ Compose multiple Criterions and calculate loss one by one.
    """
    def __init__(self, criterions, weights=None, tags=None):
        self.criterions = criterions
        self.weights = weights
        self.tags = tags

    def __call__(self, inputs, targets):
        losses = []

        for criterion, input, target in zip(self.criterions, inputs, targets):
            losses.append(criterion(input, target))

        if self.weights is not None:
            losses = [loss * weight for loss,
                      weight in zip(losses, self.weights)]
        return losses


class SoftCELoss(nn.Module):
    """ KD Loss Function
    """
    def __init__(self, T=1.0, alpha=1.0):
        super(SoftCELoss, self).__init__()
        self.T = T
        self.alpha = alpha

    def forward(self, logits, target, hard_target=None):
        ce_loss = soft_cross_entropy(logits, target, T=self.T)
        if hard_target is not None and self.alpha != 0.0:
            ce_loss += self.alpha*F.cross_entropy(logits, hard_target)
        return ce_loss


class CFLoss(nn.Module):
    """ Common Feature Learning Loss
        CF Loss = MMD + beta * MSE
    """
    def __init__(self, sigmas, normalized=True, beta=1.0):
        super(CFLoss, self).__init__()
        self.sigmas = sigmas
        self.normalized = normalized
        self.beta = beta

    def forward(self, ka_tensor, dummy=None):
        mmd_loss = 0.0
        mse_loss = 0.0
        for (hs, ht), (ft_, ft) in ka_tensor:
            for ht_i in ht:
                mmd_loss += calc_mmd(hs, ht_i, sigmas=self.sigmas,
                                     normalized=self.normalized)

            for i in range(len(ft_)):
                mse_loss += F.mse_loss(ft_[i], ft[i])

        return mmd_loss + self.beta*mse_loss

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
        nums = (1-inv_mask.float()).sum()

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
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class SoftFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(SoftFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = soft_cross_entropy(inputs, targets, size_average=False)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()
