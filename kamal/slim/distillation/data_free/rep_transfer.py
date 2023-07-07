import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

class HintLoss(nn.Module):
    """Convolutional regression for FitNet"""
    def __init__(self, s_shapes, t_shapes, use_relu=False, loss_fn=F.mse_loss):
        super(HintLoss, self).__init__()
        self.use_relu = use_relu
        self.loss_fn = loss_fn
        regs = []
        for s_shape, t_shape in zip(s_shapes, t_shapes):
            s_N, s_C, s_H, s_W = s_shape
            t_N, t_C, t_H, t_W = t_shape
            if s_H == 2 * t_H:
                conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
            elif s_H * 2 == t_H:
                conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1)
            elif s_H >= t_H:
                conv = nn.Conv2d(s_C, t_C, kernel_size=(1+s_H-t_H, 1+s_W-t_W))
            else:
                raise NotImplemented('student size {}, teacher size {}'.format(s_H, t_H))
            reg = [conv, nn.BatchNorm2d(t_C)]
            if use_relu:
                reg.append( nn.ReLU(inplace=True) )
            regs.append(nn.Sequential(*reg))
        self.regs = nn.ModuleList(regs)
        
    def forward(self, s_features, t_features):
        loss = []
        for reg, s_feat, t_feat in zip(self.regs, s_features, t_features):
            s_feat = reg(s_feat)
            loss.append( self.loss_fn( s_feat, t_feat ) )
        return loss


class ABLoss(nn.Module):
    """Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons
    code: https://github.com/bhheo/AB_distillation
    """
    def __init__(self, s_shapes, t_shapes, margin=1.0, use_relu=False):
        super(ABLoss, self).__init__()

        regs = []
        for s_shape, t_shape in zip(s_shapes, t_shapes):
            s_N, s_C, s_H, s_W = s_shape
            t_N, t_C, t_H, t_W = t_shape
            if s_H == 2 * t_H:
                conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
            elif s_H * 2 == t_H:
                conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1)
            elif s_H >= t_H:
                conv = nn.Conv2d(s_C, t_C, kernel_size=(1+s_H-t_H, 1+s_W-t_W))
            else:
                raise NotImplemented('student size {}, teacher size {}'.format(s_H, t_H))
            reg = [conv, nn.BatchNorm2d(t_C)]
            if use_relu:
                reg.append( nn.ReLU(inplace=True) )
            regs.append(nn.Sequential(*reg))
        self.regs = nn.ModuleList(regs)
        feat_num = len(self.regs)
        self.w = [2**(i-feat_num+1) for i in range(feat_num)]
        self.margin = margin

    def forward(self, s_features, t_features, reverse=False):
        s_features = [ reg(s_feat) for (reg, s_feat) in zip(self.regs, s_features) ]
        bsz = s_features[0].shape[0]
        losses = [self.criterion_alternative_l2(s, t, reverse=reverse) for s, t in zip(s_features, t_features)]
        losses = [w * l for w, l in zip(self.w, losses)]
        losses = [l / bsz for l in losses]
        losses = [l / 1000 * 3 for l in losses]
        return losses

    def criterion_alternative_l2(self, source, target, reverse):
        if reverse:
            loss = ((source - self.margin) ** 2 * ((source < self.margin) & (target <= 0)).float() +
                    (source + self.margin) ** 2 * ((source > -self.margin) & (target > 0)).float() +
                    (target - self.margin) ** 2 * ((target < self.margin) & (source <= 0)).float() +
                    (target + self.margin) ** 2 * ((target > -self.margin) & (source > 0)).float())
        else:
            loss = ((source + self.margin) ** 2 * ((source > -self.margin) & (target <= 0)).float() +
                    (source - self.margin) ** 2 * ((source <= self.margin) & (target > 0)).float()) 
        return torch.abs(loss).sum()


class RKDLoss(nn.Module):
    """Relational Knowledge Disitllation, CVPR2019"""
    def __init__(self, w_d=25, w_a=50, angle=True):
        super(RKDLoss, self).__init__()
        self.w_d = w_d
        self.w_a = w_a
        self.angle = angle

    def forward(self, s_features, t_features):
        losses = []
        for f_s, f_t in zip(s_features, t_features):
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

            if self.angle:
                # RKD Angle loss
                with torch.no_grad():
                    td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
                    norm_td = F.normalize(td, p=2, dim=2)
                    t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

                sd = (student.unsqueeze(0) - student.unsqueeze(1))
                norm_sd = F.normalize(sd, p=2, dim=2)
                s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

                loss_a = F.smooth_l1_loss(s_angle, t_angle)
            else:
                loss_a = 0
            loss = self.w_d * loss_d + self.w_a * loss_a
            losses.append(loss)
        return losses

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


class FSP(nn.Module):
    """A Gift from Knowledge Distillation:
    Fast Optimization, Network Minimization and Transfer Learning"""
    def __init__(self, s_shapes, t_shapes):
        super(FSP, self).__init__()
        assert len(s_shapes) == len(t_shapes), 'unequal length of feat list'
        s_c = [s[1] for s in s_shapes]
        t_c = [t[1] for t in t_shapes]
        if np.any(np.asarray(s_c) != np.asarray(t_c)):
            raise ValueError('num of channels not equal (error in FSP)')

    def forward(self, g_s, g_t):
        s_fsp = self.compute_fsp(g_s)
        t_fsp = self.compute_fsp(g_t)
        loss_group = [self.compute_loss(s, t) for s, t in zip(s_fsp, t_fsp)]
        return loss_group

    @staticmethod
    def compute_loss(s, t):
        return (s - t).pow(2).mean()

    @staticmethod
    def compute_fsp(g):
        fsp_list = []
        for i in range(len(g) - 1):
            bot, top = g[i], g[i + 1]
            b_H, t_H = bot.shape[2], top.shape[2]
            if b_H > t_H:
                bot = F.adaptive_avg_pool2d(bot, (t_H, t_H))
            elif b_H < t_H:
                top = F.adaptive_avg_pool2d(top, (b_H, b_H))
            else:
                pass
            bot = bot.unsqueeze(1)
            top = top.unsqueeze(2)
            bot = bot.view(bot.shape[0], bot.shape[1], bot.shape[2], -1)
            top = top.view(top.shape[0], top.shape[1], top.shape[2], -1)

            fsp = (bot * top).mean(-1)
            fsp_list.append(fsp)
        return fsp_list