# coding:utf-8
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import torch

import math
from collections import OrderedDict


class AlexNet(nn.Module):

    def __init__(self, init_weights=False, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # conv1
            nn.ReLU(inplace=False),  # relu1
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool1
            nn.Conv2d(64, 192, kernel_size=5, padding=2),  # conv2
            nn.ReLU(inplace=False),  # relu2
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2
            nn.Conv2d(192, 384, kernel_size=3, padding=1),  # conv3
            nn.ReLU(inplace=False),  # relu3
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # conv4
            nn.ReLU(inplace=False),  # relu4
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=False),
            nn.Linear(4096, num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        params = torch.load('models/alexnet-owt-4df8aa71.tar')
        del params['classifier.6.weight']
        del params['classifier.6.bias']
        params['classifier.6.weight'] = torch.zeros(
            kwargs['num_classes'], 4096).normal_(0, 0.01)
        params['classifier.6.bias'] = torch.zeros(kwargs['num_classes'])
        model.load_state_dict(params)
    return model


layer_names = [
    'conv1', 'conv1_0', 'relu1', 'pool1',
    'conv2', 'conv2_0', 'relu2', 'pool2',
    'conv3', 'conv3_0', 'relu3',
    'conv4', 'conv4_0', 'relu4',
    'conv5', 'conv5_0', 'relu5', 'pool5',
]


fc_layer_names = [
    'drop1', 'fc1', 'fc1_0', 'relu1',
    'drop2', 'fc2', 'fc2_0', 'relu2',
    'fc3'
]


compute_layer_names = [
    'conv1', 'conv2', 'conv3', 'conv4', 'conv5'
]

fc_compute_layer_names = [
    'fc1', 'fc2',
    'fc3',
]


class PartAlexNet(nn.Module):

    def __init__(self, layer_name, num_classes=1000, init_weights=False):
        super(PartAlexNet, self).__init__()

        self.layer_name = layer_name

        feature_layers = [
            ('0', nn.Conv2d(3, 72, kernel_size=11, stride=4, padding=2)),
            # ---------------------------------------------------------
            ('0_0', nn.Conv2d(72, 72, kernel_size=1)),
            # ('1', nn.ReLU(inplace=True)),
            # ('1', nn.LeakyReLU(inplace=True)),
            ('1', nn.PReLU(num_parameters=72, init=0.25)),
            ('2', nn.MaxPool2d(kernel_size=3, stride=2)),
            # ('2', nn.AvgPool2d(kernel_size=3, stride=2)),
            ('3', nn.Conv2d(72, 210, kernel_size=5, padding=2)),
            # ---------------------------------------------------------
            ('3_0', nn.Conv2d(210, 210, kernel_size=1)),
            # ('4', nn.ReLU(inplace=True)),
            # ('4', nn.LeakyReLU(inplace=True)),
            ('4', nn.PReLU(num_parameters=210, init=0.25)),
            ('5', nn.MaxPool2d(kernel_size=3, stride=2)),
            # ('5', nn.AvgPool2d(kernel_size=3, stride=2)),
            ('6', nn.Conv2d(210, 420, kernel_size=3, padding=1)),
            # ---------------------------------------------------------
            ('6_0', nn.Conv2d(420, 420, kernel_size=1)),
            # ('7', nn.ReLU(inplace=True)),
            # ('7', nn.LeakyReLU(inplace=True)),
            ('7', nn.PReLU(num_parameters=420, init=0.25)),
            ('8', nn.Conv2d(420, 320, kernel_size=3, padding=1)),
            # ---------------------------------------------------------
            ('8_0', nn.Conv2d(320, 320, kernel_size=1)),
            # ('9', nn.ReLU(inplace=True)),
            # ('9', nn.LeakyReLU(inplace=True)),
            ('9', nn.PReLU(num_parameters=320, init=1.0)),
            ('10', nn.Conv2d(320, 320, kernel_size=3, padding=1)),
            # ---------------------------------------------------------
            ('10_0', nn.Conv2d(320, 320, kernel_size=1)),
            # ('11', nn.ReLU(inplace=True)),
            # ('11', nn.LeakyReLU(inplace=True)),
            ('11', nn.PReLU(num_parameters=320, init=1.0)),
            ('12', nn.MaxPool2d(kernel_size=3, stride=2)),
            # ('12', nn.AvgPool2d(kernel_size=3, stride=2)),
        ]

        fc_layers = [
            ('0', nn.Dropout()),
            ('1', nn.Linear(320 * 6 * 6, 4200)),
            ('1_0', nn.Linear(4200, 4200)),
            # ('2', nn.ReLU(inplace=True)),
            # ('2', nn.LeakyReLU(inplace=True)),
            ('2', nn.PReLU(num_parameters=1, init=1.0)),
            ('3', nn.Dropout()),
            ('4', nn.Linear(4200, 4200)),
            ('4_0', nn.Linear(4200, 4200)),
            # ('5', nn.ReLU(inplace=True)),
            # ('5', nn.LeakyReLU(inplace=True)),
            ('5', nn.PReLU(num_parameters=1, init=1.0)),
            ('6', nn.Linear(4200, num_classes)),
        ]

        if self.layer_name[0:4] == 'conv':
            pre_idx = compute_layer_names.index(layer_name)
            if pre_idx > 0:
                pre_idx -= 1
                pre_layer_name = compute_layer_names[pre_idx]
                start_idx = layer_names.index(pre_layer_name)

                idx = layer_names.index(layer_name)

                part_feat_layers = feature_layers[start_idx+1:idx+1]
            else:
                part_feat_layers = [feature_layers[0]]
            self.features = nn.Sequential(OrderedDict(part_feat_layers))

        elif self.layer_name[0:2] == 'fc':
            if layer_name == 'fc1':
                part_feat_layers = feature_layers[-3:]
                self.features = nn.Sequential(OrderedDict(part_feat_layers))

            pre_idx = fc_compute_layer_names.index(layer_name)
            if pre_idx > 0:
                pre_idx -= 1
                pre_layer_name = fc_compute_layer_names[pre_idx]
                start_idx = fc_layer_names.index(pre_layer_name)

                idx = fc_layer_names.index(layer_name)

                part_layers = fc_layers[start_idx+1:idx+1]
            else:
                part_layers = fc_layers[0:2]
            self.classifier = nn.Sequential(OrderedDict(part_layers))

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        if self.layer_name[0:2] != 'fc':
            x = self.features(x)
        elif self.layer_name == 'fc1':
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        else:
            x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


'''
layer_names = [
    'conv1', 'relu1', 'pool1', 
    'conv2', 'relu2', 'pool2', 
    'conv3', 'relu3', 
    'conv4', 'relu4', 
    'conv5', 'relu5', 'pool5', 
    ]

fc_layer_names = [
    'drop1', 'fc1', 'relu1', 
    'drop2', 'fc2', 'relu2', 
    'fc3'
    ]

compute_layer_names = [
    'conv1', 'conv2', 'conv3', 'conv4', 'conv5'
    ]

fc_compute_layer_names = [
    'fc1', 'fc2',
    'fc3',
    ]

class PartAlexNet(nn.Module):

    def __init__(self, layer_name, num_classes=1000, init_weights=False):
        super(PartAlexNet, self).__init__()

        self.layer_name = layer_name

        feature_layers = [
            ('0', nn.Conv2d(3, 72, kernel_size=11, stride=4, padding=2)),
            # ('0_0', nn.Conv2d(72, 72, kernel_size=1)),
            ('1', nn.ReLU(inplace=True)),
            ('2', nn.MaxPool2d(kernel_size=3, stride=2)),
            ('3', nn.Conv2d(72, 210, kernel_size=5, padding=2)),
            # ('3_0', nn.Conv2d(210, 210, kernel_size=1)),
            ('4', nn.ReLU(inplace=True)),
            ('5', nn.MaxPool2d(kernel_size=3, stride=2)),
            ('6', nn.Conv2d(210, 420, kernel_size=3, padding=1)),
            # ('6_0', nn.Conv2d(420, 420, kernel_size=1)),
            ('7', nn.ReLU(inplace=True)),
            ('8', nn.Conv2d(420, 320, kernel_size=3, padding=1)),
            # ('8_0', nn.Conv2d(320, 320, kernel_size=1)),
            ('9', nn.ReLU(inplace=True)),
            ('10', nn.Conv2d(320, 320, kernel_size=3, padding=1)),
            # ('10_0', nn.Conv2d(320, 320, kernel_size=1)),
            ('11', nn.ReLU(inplace=True)),
            ('12', nn.MaxPool2d(kernel_size=3, stride=2)),
        ]

        fc_layers = [
            ('0', nn.Dropout()),
            ('1', nn.Linear(320 * 6 * 6, 4200)),
            # ('1_0', nn.Linear(4200, 4200)),
            ('2', nn.ReLU(inplace=True)),
            ('3', nn.Dropout()),
            ('4', nn.Linear(4200, 4200)),
            # ('4_0', nn.Linear(4200, 4200)),
            ('5', nn.ReLU(inplace=True)),
            ('6', nn.Linear(4200, num_classes)),
        ]

        if self.layer_name[0:4] == 'conv':
            pre_idx = compute_layer_names.index(layer_name)
            if pre_idx > 0:
                pre_idx -= 1
                pre_layer_name = compute_layer_names[pre_idx]
                start_idx = layer_names.index(pre_layer_name)

                idx = layer_names.index(layer_name)

                part_feat_layers = feature_layers[start_idx+1:idx+1]
            else:
                part_feat_layers = [feature_layers[0]]
            self.features = nn.Sequential(OrderedDict(part_feat_layers))

        elif self.layer_name[0:2] == 'fc':
            if layer_name == 'fc1':
                # part_feat_layers = feature_layers[-3:]
                part_feat_layers = feature_layers[-2:]
                self.features = nn.Sequential(OrderedDict(part_feat_layers))

            pre_idx = fc_compute_layer_names.index(layer_name)
            if pre_idx > 0:
                pre_idx -= 1
                pre_layer_name = fc_compute_layer_names[pre_idx]
                start_idx = fc_layer_names.index(pre_layer_name)

                idx = fc_layer_names.index(layer_name)

                part_layers = fc_layers[start_idx+1:idx+1]
            else:
                part_layers = fc_layers[0:2]
                # part_layers = fc_layers[0:1]
            self.classifier = nn.Sequential(OrderedDict(part_layers))

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # x = self.features(x)
        # x = x.view(x.size(0), 320 * 6 * 6)
        # x = self.classifier(x)
        if self.layer_name[0:2] != 'fc':
            x = self.features(x)
        elif self.layer_name == 'fc1':
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        else:
            x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
'''


def part_alexnet(layer_name, pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = PartAlexNet(layer_name, **kwargs)
    return model


class DistilledAlexNet(nn.Module):

    def __init__(self, num_classes=1000, init_weights=False):
        super(DistilledAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 72, kernel_size=11, stride=4, padding=2),  # conv1
            # nn.ReLU(inplace=False), # relu1
            nn.PReLU(num_parameters=72, init=0.25),
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool1
            nn.Conv2d(72, 210, kernel_size=5, padding=2),  # conv2
            # nn.ReLU(inplace=False), # relu2
            nn.PReLU(num_parameters=210, init=0.25),
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2
            nn.Conv2d(210, 420, kernel_size=3, padding=1),  # conv3
            # nn.ReLU(inplace=False), # relu3
            # nn.LeakyReLU(inplace=True),
            nn.PReLU(num_parameters=420, init=0.25),
            nn.Conv2d(420, 320, kernel_size=3, padding=1),  # conv4
            # nn.ReLU(inplace=False), # relu4
            # nn.LeakyReLU(inplace=True),
            nn.PReLU(num_parameters=320, init=1.0),
            nn.Conv2d(320, 320, kernel_size=3, padding=1),
            # nn.ReLU(inplace=False),
            # nn.LeakyReLU(inplace=True),
            nn.PReLU(num_parameters=320, init=1.0),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(320 * 6 * 6, 4200),
            # nn.ReLU(inplace=False),
            nn.PReLU(num_parameters=1, init=1.0),
            nn.Dropout(),
            nn.Linear(4200, 4200),
            # nn.ReLU(inplace=False),
            nn.PReLU(num_parameters=1, init=1.0),
            nn.Linear(4200, num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def distilled_alexnet(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DistilledAlexNet(**kwargs)
    return model


class Distilled_Full_AlexNet(nn.Module):

    def __init__(self, num_classes=1000, init_weights=False):
        super(Distilled_Full_AlexNet, self).__init__()
        self.features = nn.Sequential(
            OrderedDict([
                ('0', nn.Conv2d(3, 72, kernel_size=11, stride=4, padding=2)),
                ('0_0', nn.Conv2d(72, 72, kernel_size=1)),
                # ('1', nn.ReLU(inplace=True)),
                ('1', nn.PReLU(num_parameters=72, init=0.25)),
                ('2', nn.MaxPool2d(kernel_size=3, stride=2)),
                ('3', nn.Conv2d(72, 210, kernel_size=5, padding=2)),
                ('3_0', nn.Conv2d(210, 210, kernel_size=1)),
                # ('4', nn.ReLU(inplace=True)),
                ('4', nn.PReLU(num_parameters=210, init=0.25)),
                ('5', nn.MaxPool2d(kernel_size=3, stride=2)),
                ('6', nn.Conv2d(210, 420, kernel_size=3, padding=1)),
                ('6_0', nn.Conv2d(420, 420, kernel_size=1)),
                # ('7', nn.ReLU(inplace=True)),
                # ('7', nn.LeakyReLU(inplace=True)),
                ('7', nn.PReLU(num_parameters=420, init=0.25)),
                ('8', nn.Conv2d(420, 320, kernel_size=3, padding=1)),
                ('8_0', nn.Conv2d(320, 320, kernel_size=1)),
                # ('9', nn.ReLU(inplace=True)),
                # ('9', nn.LeakyReLU(inplace=True)),
                ('9', nn.PReLU(num_parameters=320, init=1.0)),
                ('10', nn.Conv2d(320, 320, kernel_size=3, padding=1)),
                ('10_0', nn.Conv2d(320, 320, kernel_size=1)),
                # ('11', nn.ReLU(inplace=True)),
                # ('11', nn.LeakyReLU(inplace=True)),
                ('11', nn.PReLU(num_parameters=320, init=1.0)),
                ('12', nn.MaxPool2d(kernel_size=3, stride=2)),
            ])
        )
        self.classifier = nn.Sequential(
            OrderedDict([
                ('0', nn.Dropout()),
                ('1', nn.Linear(320 * 6 * 6, 4200)),
                ('1_0', nn.Linear(4200, 4200)),
                # ('2', nn.ReLU(inplace=True)),
                ('2', nn.PReLU(num_parameters=1, init=1.0)),
                ('3', nn.Dropout()),
                ('4', nn.Linear(4200, 4200)),
                ('4_0', nn.Linear(4200, 4200)),
                # ('5', nn.ReLU(inplace=True)),
                ('5', nn.PReLU(num_parameters=1, init=1.0)),
                ('6', nn.Linear(4200, num_classes)),
            ])
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def distilled_full_alexnet(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Distilled_Full_AlexNet(**kwargs)
    return model
