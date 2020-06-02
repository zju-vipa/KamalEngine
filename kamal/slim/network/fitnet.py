import torch.nn as nn

class Regressor(nn.Module):
    """Convolutional regression for FitNet"""
    def __init__(self, s_shape, t_shape, is_relu=True):
        super(Regressor, self).__init__()
        self.is_relu = is_relu
        _, s_C, s_H, s_W = s_shape
        _, t_C, t_H, t_W = t_shape
        if s_H == 2 * t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1)
        elif s_H >= t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=(1+s_H-t_H, 1+s_W-t_W))
        else:
            raise NotImplemented('student size {}, teacher size {}'.format(s_H, t_H))
        self.bn = nn.BatchNorm2d(t_C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.is_relu:
            return self.relu(self.bn(x))
        else:
            return self.bn(x)
