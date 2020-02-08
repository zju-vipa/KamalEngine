# This implementation is based on the DenseNet-BC implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from copy import deepcopy


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                        kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
        efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
    """
    def __init__(self, growth_rate=12, block_config=(16, 16, 16), compression=0.5,
                 num_init_features=24, bn_size=4, drop_rate=0,
                 num_classes=10, small_inputs=True, efficient=False):

        super(DenseNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'

        # First convolution
        if small_inputs:
            self.preprocess = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
        else:
            self.preprocess = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ]))
            self.preprocess.add_module('norm0', nn.BatchNorm2d(num_init_features))
            self.preprocess.add_module('relu0', nn.ReLU(inplace=True))
            self.preprocess.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                                           ceil_mode=False))

        # Each denseblock
        num_features = num_init_features
        self.denseblocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.denseblocks.append(block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.transitions.append(trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.norm_final = nn.BatchNorm2d(num_features)

        # Linear layer
        self.fc = nn.Linear(num_features, num_classes)

        # Initialization
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                n = param.size(0) * param.size(2) * param.size(3)
                param.data.normal_().mul_(math.sqrt(2. / n))
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)
            elif 'fc' in name and 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        features = self.preprocess(x)
        for i in range(len(self.denseblocks)):
            features = self.denseblocks[i](features)
            if i < len(self.transitions):
                features = self.transitions[i](features)
        features = self.norm_final(features)

        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.fc(out)
        return out


class JointDenseNet(nn.Module):
    def __init__(self, teachers, indices, phase):
        super(JointDenseNet, self).__init__()
        assert(len(indices) == len(teachers))
        self.indices = indices
        self.phase = phase
        self.preprocess = deepcopy(teachers[0].preprocess)
        self.denseblocks = deepcopy(teachers[0].denseblocks)
        self.transitions = deepcopy(teachers[0].transitions)

        # Initialization before copying modules and parameters of teachers.
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                n = param.size(0) * param.size(2) * param.size(3)
                param.data.normal_().mul_(math.sqrt(2. / n))
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)

        self.norm_finals = nn.ModuleList([deepcopy(teacher.norm_final) for teacher in teachers])
        self.fcs = nn.ModuleList([deepcopy(teacher.fc) for teacher in teachers])

        self.teacher_denseblocks_list = nn.ModuleList([deepcopy(teacher.denseblocks) for teacher in teachers])
        self.teacher_transitions_list = nn.ModuleList([deepcopy(teacher.transitions) for teacher in teachers])

        # Whether to fix parameters of branches from teachers when training each block.
        for name, param in self.teacher_denseblocks_list.named_parameters():
            param.requires_grad = (self.phase != 'block')
        for name, param in self.teacher_transitions_list.named_parameters():
            param.requires_grad = (self.phase != 'block')
        for name, param in self.norm_finals.named_parameters():
            param.requires_grad = (self.phase != 'block')
        for name, param in self.fcs.named_parameters():
            param.requires_grad = (self.phase != 'block')

    def forward(self, x):
        num_b = len(self.denseblocks)
        x = self.preprocess(x)
        features_list = [None for i in range(len(self.indices))]

        out_idx = max(self.indices)
        for i in range(out_idx):
            x = self.denseblocks[i](x)
            if i < num_b-1:
                x = self.transitions[i](x)
            for j in range(len(self.indices)):
                if i == self.indices[j]-1:
                    features_list[j] = x

        # Mimic teachers.
        for i in range(len(self.indices)):
            for j in range(self.indices[i], num_b):
                features_list[i] = self.teacher_denseblocks_list[i][j](features_list[i])
                if j < num_b-1:
                    features_list[i] = self.teacher_transitions_list[i][j](features_list[i])

        features_list = [self.norm_finals[i](f) for i,f in enumerate(features_list)]
        outs = [F.relu(f, inplace=True) for f in features_list]
        outs = [F.adaptive_avg_pool2d(out, (1,1)).view(features_list[0].size(0), -1) for out in outs]
        outs = [self.fcs[i](outs[i]) for i in range(len(self.indices))]
        out = torch.cat(outs, dim=1)

        return out