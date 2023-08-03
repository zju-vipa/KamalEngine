import torch.nn as nn
from models.util import ConvReg


# class HintLoss(nn.Module):
#     """Fitnets: hints for thin deep nets, ICLR 2015"""
#     def __init__(self, conv_reg: ConvReg, hint_layer: int):
#         super(HintLoss, self).__init__()
#         self.crit = nn.MSELoss()
#         self.conv_reg = conv_reg
#         self.hint_layer = hint_layer
#
#     def forward(self, f_s, f_t):
#         f_s = self.conv_reg(f_s[self.hint_layer])
#         f_t = f_t[self.hint_layer]
#         loss = self.crit(f_s, f_t)
#         return loss


class HintLoss(nn.Module):
    """Fitnets: hints for thin deep nets, ICLR 2015"""
    def __init__(self):
        super(HintLoss, self).__init__()
        self.crit = nn.MSELoss()

    def forward(self, f_s, f_t):
        loss = self.crit(f_s, f_t)
        return loss