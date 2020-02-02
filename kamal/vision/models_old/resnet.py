import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
from copy import deepcopy
from collections import OrderedDict
import numpy as np
from functools import reduce


__all__ = ['ResNet', 'JointResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Attention(nn.Module):
    def __init__(self, in_channel, out_channel, r=8):
        super(Attention, self).__init__()
        se = out_channel / r
        se = 16 if se < 16 else se
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channel, se, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(se, out_channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.attention(x)

class JointBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, num_stream, stride=1, downsample=None):
        super(JointBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.attention_modules = nn.ModuleList([Attention(planes, planes * 4) for i in range(num_stream)])

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride
        self.num_stream = num_stream
        self.cnt = 0

    def forward(self, xs):
        assert(self.num_stream == len(xs))
        residuals = xs
        if self.downsample is not None:
            residuals = [self.downsample(x) for x in xs]

        outs = [self.conv1(x) for x in xs]
        outs = [self.bn1(out) for out in outs]
        outs = [self.relu(out) for out in outs]

        outs = [self.conv2(out) for out in outs]
        outs = [self.bn2(out) for out in outs]
        outs = [self.relu(out) for out in outs]

        atts = [self.attention_modules[i](outs[i]) for i in range(len(xs))]

        outs = [self.conv3(out) for out in outs]
        outs = [self.bn3(out) for out in outs]

        outs = [outs[i] + residuals[i] * atts[i] for i in range(len(xs))]
        outs = [self.relu(out) for out in outs]


        return outs

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class JointResNet(nn.Module):
    def __init__(self, teachers, layers, indices, phase):
        super(JointResNet, self).__init__()
        # Only support bottleneck block by now.
        assert(len(teachers) == len(indices))

        self.teacher_layers_list = nn.ModuleList()
        self.teacher_fcs = nn.ModuleList()
        for teacher in teachers:
            self.teacher_layers_list.append(nn.ModuleList(deepcopy(list(teacher.children())[4:8])))
            self.teacher_fcs.append(deepcopy(teacher.fc))

        self.indices = indices
        self.num_stream = len(teachers)

        self.phase = phase
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = nn.ModuleList()
        self.layers.append(nn.ModuleList([self._make_layer(JointBottleneck, 64, 1) for i in range(layers[0])]))
        self.layers.append(nn.ModuleList([self._make_layer(JointBottleneck, 128, 1, stride=2 if i == 0 else 1) for i in range(layers[1])]))
        self.layers.append(nn.ModuleList([self._make_layer(JointBottleneck, 256, 1, stride=2 if i == 0 else 1) for i in range(layers[2])]))
        self.layers.append(nn.ModuleList([self._make_layer(JointBottleneck, 512, 1, stride=2 if i == 0 else 1) for i in range(layers[3])]))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Whether to fix parameters of branches from teachers when training each block.
        for name, param in self.teacher_layers_list.named_parameters():
            param.requires_grad = (self.phase != 'block')
        for name, param in self.teacher_fcs.named_parameters():
            param.requires_grad = (self.phase != 'block')

        # Initialize params.
        for name, child in self.named_children():
            if 'teacher' not in name:
                for m in child.modules():
                    if isinstance(m, nn.Conv2d):
                        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                        m.weight.data.normal_(0, math.sqrt(2. / n))
                    elif isinstance(m, nn.BatchNorm2d):
                        m.weight.data.fill_(1)
                        m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, num_stream=self.num_stream, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, num_stream=self.num_stream))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        xs = []

        out_features = [None for i in range(self.num_stream)]
        xs = [x for i in range(self.num_stream)]
        k = 0
        flag = False
        for i in range(4):
            for module in self.layers[i]:
                for j in range(self.num_stream):
                    if k == self.indices[j]:
                        out_features[j] = xs[j]
                if k == max(self.indices):
                    flag = True
                    break
                xs = module(xs)
                k += 1        
            if flag:
                    break

        for i in range(self.num_stream):
            if k == self.indices[i] :
                out_features[i] = xs[i]     

        # Mimic teachers
        xs = out_features
        for i in range(self.num_stream):
            x = out_features[i]
            j = 0
            for layer in self.teacher_layers_list[i]:
                for name, module in layer.named_children():
                    if j >= self.indices[i]:
                        x = module(x)
                    j += 1
            xs[i] = x

        xs = [self.avgpool(x) for x in xs]
        xs = [x.view(x.size(0), -1) for x in xs]
        xs = [self.teacher_fcs[i](x) for i, x in enumerate(xs)]
        x = torch.cat(xs, dim=1)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1000)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    num_classes = kwargs.get('num_classes', None)
    if num_classes is not None and num_classes != 1000:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=1000)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    num_classes = kwargs.get('num_classes', None)
    if num_classes is not None and num_classes != 1000:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=1000)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    num_classes = kwargs.get('num_classes', None)
    if num_classes is not None and num_classes != 1000:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=1000)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    num_classes = kwargs.get('num_classes', None)
    if num_classes is not None and num_classes != 1000:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes=1000)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    num_classes = kwargs.get('num_classes', None)
    if num_classes is not None and num_classes != 1000:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
