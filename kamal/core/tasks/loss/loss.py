# Copyright 2020 Zhejiang Lab. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================
import json
import logging
from typing import Dict, List

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Module, ModuleDict
import numpy as np
#from pytorch_msssim import ssim, ms_ssim, MS_SSIM, SSIM

from .functional import *

class KLDiv(object):
    def __init__(self, T=1.0):
        self.T = T
    
    def __call__(self, logits, targets):
        return kldiv( logits, targets, T=self.T )
class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction="batchmean") * (self.T**2)
        return loss
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
        CFL Loss = MMD + MSE
    """
    def __init__(self, sigmas, normalized=True):
        super(CFLLoss, self).__init__()
        self.sigmas = sigmas
        self.normalized = normalized

    def forward(self, hs, hts, fts_, fts):
        mmd = mse = 0.0
        for ht_i in hts:
            mmd += mmd_loss(hs, ht_i, sigmas=self.sigmas, normalized=self.normalized)
        for i in range(len(fts_)):
            mse += F.mse_loss(fts_[i], fts[i])
        return mmd, mse

class PSNR_Loss(nn.Module):
    def __init__(self, data_range=1.0, size_average=True):
        super(PSNR_Loss, self).__init__()
        self.data_range = data_range
        self.size_average = size_average

    def forward(self, img1, img2):
        return 100 - psnr(img1, img2, size_average=self.size_average, data_range=self.data_range)


#class MS_SSIM_Loss(MS_SSIM):
#    def forward(self, img1, img2):
#        return 100*(1 - super(MS_SSIM_Loss, self).forward(img1, img2))

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

class AttentionLoss(nn.Module):
    """ Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer"""
    def __init__(self, p=2):
        super(AttentionLoss, self).__init__()
        self.p = p

    def forward(self, g_s, g_t):
        return sum([self.at_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)])

    def at_loss(self, f_s, f_t):
        s_H, t_H = f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
        else:
            pass
        return (self.at(f_s) - self.at(f_t)).pow(2).mean()

    def at(self, f):
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))

class NSTLoss(nn.Module):
    """like what you like: knowledge distill via neuron selectivity transfer"""
    def __init__(self):
        super(NSTLoss, self).__init__()
        pass

    def forward(self, g_s, g_t):
        return sum([self.nst_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)])

    def nst_loss(self, f_s, f_t):
        s_H, t_H = f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
        else:
            pass

        f_s = f_s.view(f_s.shape[0], f_s.shape[1], -1)
        f_s = F.normalize(f_s, dim=2)
        f_t = f_t.view(f_t.shape[0], f_t.shape[1], -1)
        f_t = F.normalize(f_t, dim=2)

        # set full_loss as False to avoid unnecessary computation
        full_loss = True
        if full_loss:
            return (self.poly_kernel(f_t, f_t).mean().detach() + self.poly_kernel(f_s, f_s).mean()
                    - 2 * self.poly_kernel(f_s, f_t).mean())
        else:
            return self.poly_kernel(f_s, f_s).mean() - 2 * self.poly_kernel(f_s, f_t).mean()

    def poly_kernel(self, a, b):
        a = a.unsqueeze(1)
        b = b.unsqueeze(2)
        res = (a * b).sum(-1).pow(2)
        return res

class SPLoss(nn.Module):
    """Similarity-Preserving Knowledge Distillation, ICCV2019, verified by original author"""
    def __init__(self):
        super(SPLoss, self).__init__()

    def forward(self, g_s, g_t):
        return sum([self.similarity_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)])

    def similarity_loss(self, f_s, f_t):
        bsz = f_s.shape[0]
        f_s = f_s.view(bsz, -1)
        f_t = f_t.view(bsz, -1)

        G_s = torch.mm(f_s, torch.t(f_s))
        # G_s = G_s / G_s.norm(2)
        G_s = torch.nn.functional.normalize(G_s)
        G_t = torch.mm(f_t, torch.t(f_t))
        # G_t = G_t / G_t.norm(2)
        G_t = torch.nn.functional.normalize(G_t)

        G_diff = G_t - G_s
        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
        return loss

class RKDLoss(nn.Module):
    """Relational Knowledge Disitllation, CVPR2019"""
    def __init__(self, w_d=25, w_a=50):
        super(RKDLoss, self).__init__()
        self.w_d = w_d
        self.w_a = w_a

    def forward(self, f_s, f_t):
        student = f_s.view(f_s.shape[0], -1)
        teacher = f_t.view(f_t.shape[0], -1)

        # RKD distance loss
        with torch.no_grad():
            t_d = self.pdist(teacher, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = self.pdist(student, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        loss_d = F.smooth_l1_loss(d, t_d)

        # RKD Angle loss
        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss_a = F.smooth_l1_loss(s_angle, t_angle)

        loss = self.w_d * loss_d + self.w_a * loss_a

        return loss

    @staticmethod
    def pdist(e, squared=False, eps=1e-12):
        e_square = e.pow(2).sum(dim=1)
        prod = e @ e.t()
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

        if not squared:
            res = res.sqrt()

        res = res.clone()
        res[range(len(e)), range(len(e))] = 0
        return res

class PKTLoss(nn.Module):
    """Probabilistic Knowledge Transfer for deep representation learning"""
    def __init__(self):
        super(PKTLoss, self).__init__()

    def forward(self, f_s, f_t):
        return self.cosine_similarity_loss(f_s, f_t)

    @staticmethod
    def cosine_similarity_loss(output_net, target_net, eps=0.0000001):
        # Normalize each vector by its norm
        output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
        output_net = output_net / (output_net_norm + eps)
        output_net[output_net != output_net] = 0

        target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
        target_net = target_net / (target_net_norm + eps)
        target_net[target_net != target_net] = 0

        # Calculate the cosine similarity
        model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
        target_similarity = torch.mm(target_net, target_net.transpose(0, 1))

        # Scale cosine similarity to 0..1
        model_similarity = (model_similarity + 1.0) / 2.0
        target_similarity = (target_similarity + 1.0) / 2.0

        # Transform them into probabilities
        model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
        target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)

        # Calculate the KL-divergence
        loss = torch.mean(target_similarity * torch.log((target_similarity + eps) / (model_similarity + eps)))

        return loss

class SVDLoss(nn.Module):
    """
    Self-supervised Knowledge Distillation using Singular Value Decomposition
    """
    def __init__(self, k=1):
        super(SVDLoss, self).__init__()
        self.k = k

    def forward(self, g_s, g_t):
        v_sb = None
        v_tb = None
        losses = []
        for i, f_s, f_t in zip(range(len(g_s)), g_s, g_t):

            u_t, s_t, v_t = self.svd(f_t, self.k)
            u_s, s_s, v_s = self.svd(f_s, self.k + 3)
            v_s, v_t = self.align_rsv(v_s, v_t)
            s_t = s_t.unsqueeze(1)
            v_t = v_t * s_t
            v_s = v_s * s_t

            if i > 0:
                s_rbf = torch.exp(-(v_s.unsqueeze(2) - v_sb.unsqueeze(1)).pow(2) / 8)
                t_rbf = torch.exp(-(v_t.unsqueeze(2) - v_tb.unsqueeze(1)).pow(2) / 8)

                l2loss = (s_rbf - t_rbf.detach()).pow(2)
                l2loss = torch.where(torch.isfinite(l2loss), l2loss, torch.zeros_like(l2loss))
                losses.append(l2loss.sum())

            v_tb = v_t
            v_sb = v_s

        bsz = g_s[0].shape[0]
        losses = [l / bsz for l in losses]
        return sum(losses)

    def svd(self, feat, n=1):
        size = feat.shape
        assert len(size) == 4

        x = feat.view(-1, size[1], size[2] * size[2]).transpose(-2, -1)
        u, s, v = torch.svd(x)

        u = self.removenan(u)
        s = self.removenan(s)
        v = self.removenan(v)

        if n > 0:
            u = F.normalize(u[:, :, :n], dim=1)
            s = F.normalize(s[:, :n], dim=1)
            v = F.normalize(v[:, :, :n], dim=1)

        return u, s, v

    @staticmethod
    def removenan(x):
        x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        return x

    @staticmethod
    def align_rsv(a, b):
        cosine = torch.matmul(a.transpose(-2, -1), b)
        max_abs_cosine, _ = torch.max(torch.abs(cosine), 1, keepdim=True)
        mask = torch.where(torch.eq(max_abs_cosine, torch.abs(cosine)),
                           torch.sign(cosine), torch.zeros_like(cosine))
        a = torch.matmul(a, mask)
        return a, b
        
class ABLoss(nn.Module):
    """Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons
    code: https://github.com/bhheo/AB_distillation
    """
    def __init__(self, feat_num, margin=1.0):
        super(ABLoss, self).__init__()
        self.w = [2**(i - feat_num + 1) for i in range(feat_num)]
        self.margin = margin

    def forward(self, g_s, g_t):
        bsz = g_s[0].shape[0]
        losses = [self.criterion_alternative_l2(s, t) for s, t in zip(g_s, g_t)]
        losses = [w * l for w, l in zip(self.w, losses)]
        # loss = sum(losses) / bsz
        # loss = loss / 1000 * 3
        losses = [l / bsz for l in losses]
        losses = [l / 1000 * 3 for l in losses]
        return losses

    def criterion_alternative_l2(self, source, target):
        loss = ((source + self.margin) ** 2 * ((source > -self.margin) & (target <= 0)).float() +
                (source - self.margin) ** 2 * ((source <= self.margin) & (target > 0)).float())
        return torch.abs(loss).sum()
    


class Correlation(nn.Module):
    """Correlation Congruence for Knowledge Distillation, ICCV 2019.
    The authors nicely shared the code with me. I restructured their code to be
    compatible with my running framework. Credits go to the original author"""
    def __init__(self):
        super(Correlation, self).__init__()

    def forward(self, f_s, f_t):
        delta = torch.abs(f_s - f_t)
        loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
        return loss

class HintLoss(nn.Module):
    """Fitnets: hints for thin deep nets, ICLR 2015"""
    def __init__(self, conv_reg, hint_layer):
        super(HintLoss, self).__init__()
        self.crit = nn.MSELoss()
        self.conv_reg = conv_reg
        self.hint_layer = hint_layer

    def forward(self, f_s, f_t):
        f_s = self.conv_reg(f_s[self.hint_layer])
        f_t = f_t[self.hint_layer]
        loss = self.crit(f_s, f_t)
        return loss    

class FeatureClassify(Module):
    def __init__(self, input_channels: int, num_classes: int):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(self.input_channels, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.global_avg_pool(x)
        x = x.squeeze()
        x = self.fc(x)
        return x 
class HierarchicalLoss(Module):
    def __init__(self, layer_cluster_info: Dict[str, str], feat_s: List[torch.Tensor]):
        """
        Args:
            layer_cluster_info: cluster info filepath by layer name
        """
        self.logger = logging.getLogger("HierarchicalLoss")

        super().__init__()

        # loading cluster info from merge.json files
        self.cluster_info: Dict[str, List[List[int]]] = dict()
        for name, info_file in layer_cluster_info.items():
            with open(info_file, "r") as f:
                info: List[List[int]] = json.load(f)["merged_classes"]
            self.cluster_info[name] = info

        # for each extracted layer, create classifier
        self.layer_cls_loss_fn: ModuleDict[str, nn.CrossEntropyLoss] = ModuleDict()
        layer_cls_dict: Dict[str, FeatureClassify] = dict()
        for name, info in self.cluster_info.items():
            idx = int(name[-1])
            self.logger.warning(
                "Mapping teacher layer: %s to the %d-th part of student output feature with shape: %s!!!",
                name, idx, feat_s[idx].shape
            )
            layer_cls_dict[name] = FeatureClassify(
                input_channels=feat_s[idx].shape[1],
                num_classes=len(info)
            )
            self.layer_cls_loss_fn[name] = nn.CrossEntropyLoss(weight=self._get_cluster_weight(info))

        self.feature_classifiers = ModuleDict(layer_cls_dict)
        self.layer_cluster_map = self._layer_cluster_class_map()

    def _get_cluster_weight(self, info: List[List[int]]):
        weight = torch.empty(len(info))
        for idx, cluster in enumerate(info):
            num = len(cluster)
            weight[idx] = 1 / num
        return weight

    def _layer_cluster_class_map(self) -> Dict[str, Dict[int, int]]:
        layer_cluster_map = dict()
        for name, info in self.cluster_info.items():
            cluster_map = dict()
            for cluster_id, cluster in enumerate(info):
                for fine_id in cluster:
                    cluster_map[fine_id] = cluster_id
            layer_cluster_map[name] = cluster_map
        return layer_cluster_map

    def cluster_cls_loss(
        self,
        layer_name: str,
        pred: torch.Tensor,
        target: torch.Tensor
    ):
        cluster_map = self.layer_cluster_map[layer_name]
        t = torch.empty_like(target)
        assert target.dim() == 1, "target dim {} is not equal to 1"
        for i, x in enumerate(target):
            t[i] = cluster_map[x.item()]
        loss = self.layer_cls_loss_fn[layer_name](pred, t)
        return loss

    def forward(self, feat_s: List[torch.Tensor], logit_t: torch.Tensor):
        losses: List[torch.Tensor] = list()
        for name, classifier in self.feature_classifiers.items():
            idx = int(name[-1])
            pred = classifier(feat_s[idx])
            losses.append(self.cluster_cls_loss(name, pred, logit_t))
        return sum(losses)

class VIDLoss(nn.Module):
    """Variational Information Distillation for Knowledge Transfer (CVPR 2019),
    code from author: https://github.com/ssahn0215/variational-information-distillation"""
    def __init__(
        self,
        num_input_channels,
        num_mid_channel,
        num_target_channels,
        init_pred_var=5.0,
        eps=1e-5
    ):
        super(VIDLoss, self).__init__()

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(
                in_channels, out_channels,
                kernel_size=1, padding=0,
                bias=False, stride=stride)

        self.regressor = nn.Sequential(
            conv1x1(num_input_channels, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_target_channels),
        )
        self.log_scale = torch.nn.Parameter(
            np.log(np.exp(init_pred_var - eps) - 1.0) * torch.ones(num_target_channels)
        )
        self.eps = eps

    def forward(self, input, target):
        # pool for dimentsion match
        s_H, t_H = input.shape[2], target.shape[2]
        if s_H > t_H:
            input = F.adaptive_avg_pool2d(input, (t_H, t_H))
        elif s_H < t_H:
            target = F.adaptive_avg_pool2d(target, (s_H, s_H))
        else:
            pass
        pred_mean = self.regressor(input)
        pred_var = torch.log(1.0 + torch.exp(self.log_scale)) + self.eps
        pred_var = pred_var.view(1, -1, 1, 1)
        neg_log_prob = 0.5 * (
            (pred_mean - target)**2 / pred_var + torch.log(pred_var)
        )
        loss = torch.mean(neg_log_prob)
        return loss

class MidClsLoss(Module):
    def __init__(self, num_classes: int, mid_layer_T: Dict[str, float], feat_s: List[torch.Tensor]):
        self.logger = logging.getLogger("HierarchicalLoss")
        super().__init__()

        self.layer_classifier: ModuleDict[str, FeatureClassify] = ModuleDict()
        self.layer_loss: ModuleDict[str, DistillKL] = ModuleDict()
        for name, T in mid_layer_T.items():
            idx = int(name[-1])
            self.logger.warning(
                "Mapping teacher layer: %s to the %d-th part of student output feature with shape: %s!!!",
                name, idx, feat_s[idx].shape
            )
            self.layer_classifier[name] = FeatureClassify(
                input_channels=feat_s[idx].shape[1],
                num_classes=num_classes
            )
            self.layer_loss[name] = DistillKL(T)

    def forward(self, feat_s: List[torch.Tensor], target: torch.Tensor):
        losses: List[torch.Tensor] = list()
        for name in self.layer_classifier.keys():
            idx = int(name[-1])
            pred = self.layer_classifier[name](feat_s[idx])
            losses.append(self.layer_loss[name](pred, target))
        return sum(losses)    


    
