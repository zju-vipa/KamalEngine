import torch.nn as nn
import torch.nn.functional as F
import torch
import math

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

class DistillKL_NLL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL_NLL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        #y_s = torch.pow(math.e, y_s)
        p_s = F.log_softmax(y_s/ self.T, dim=1)

        # p_t = F.softmax(y_t, dim=1)
        # p_t = torch.pow(math.e, p_t)
        p_t = F.softmax(y_t / self.T, dim=1)

        loss = F.kl_div(p_s, p_t, reduction="batchmean") * (self.T**2)
        return loss
