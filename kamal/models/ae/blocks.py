import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Function


class DiffClamp(Function):
    @staticmethod
    def forward(ctx, x, min, max):
        return x.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


class GradNormBlock(nn.Module):
    def __init__(self):
        super(GradNormBlock, self).__init__()

    def forward(self, x):
        return GradNormFunction.apply(x)


class GradNormFunction(Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        ori_shape = grad_output.shape
        return F.normalize(grad_output.view(ori_shape[0], -1), dim=1).view(ori_shape).clone()


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """ Basic Block for residual net
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class _DenseLayer(nn.Module):
    def __init__(self, inplanes, growth_rate, bn_size):
        super(_DenseLayer, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        new_features = self.relu(self.conv1(x))
        new_features = self.relu(self.conv2(new_features))
        return torch.cat((x, new_features), dim=1), new_features


class DenseBlock(nn.Module):
    """ Densely Connected Block
    """

    def __init__(self, num_layers, inplanes, bn_size, growth_rate):
        super(DenseBlock, self).__init__()
        dense_layers = []

        for i in range(num_layers):
            layer = _DenseLayer(inplanes + i * growth_rate,
                                growth_rate, bn_size)
            dense_layers.append(layer)

        self.dense_layers = nn.ModuleList(dense_layers)

    def forward(self, x):
        new_features = []
        for dense_layer in self.dense_layers:
            x, new_fea = dense_layer(x)
            new_features.append(new_fea)
        return x, new_features
