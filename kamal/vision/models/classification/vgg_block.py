from collections import OrderedDict

import torch
import torch.nn as nn


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],

    'vgg16-graft': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M'],
}


def make_layers_block_wise(cfg, block_wise, in_channels=3):
    batch_norm = False

    if block_wise:
        blocks = []
        for block_cfg in cfg:
            order = 0
            layers = []
            for v in block_cfg:
                if v == 'M':
                    layers += [(str(order), nn.MaxPool2d(kernel_size=2, stride=2))]
                    order += 1
                else:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                    if batch_norm:
                        layers += [
                            (str(order), conv2d),
                            (str(order + 1), nn.BatchNorm2d(v)),
                            (str(order + 2), nn.ReLU(inplace=True))
                        ]
                        order += 3
                    else:
                        layers += [
                            (str(order), conv2d),
                            (str(order + 1), nn.ReLU(inplace=True))
                        ]
                        order += 2

                    in_channels = v
            blocks.append(nn.Sequential(OrderedDict(layers)))

        return nn.Sequential(*blocks)

    else:
        order = 0
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [(str(order), nn.MaxPool2d(kernel_size=2, stride=2))]
                order += 1
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [
                        (str(order), conv2d),
                        (str(order + 1), nn.BatchNorm2d(v)),
                        (str(order + 2), nn.ReLU(inplace=True))
                    ]
                    order += 3
                else:
                    layers += [
                        (str(order), conv2d),
                        (str(order + 1), nn.ReLU(inplace=True))
                    ]
                    order += 2

                in_channels = v

        return nn.Sequential(OrderedDict(layers))


def split_block(cfg):
    blocks = []
    block_cur = []
    for item in cfg:
        if item != 'M':
            block_cur.append(item)
        else:
            block_cur.append(item)
            blocks.append(block_cur)
            block_cur = []

    return blocks

# ----------------- VGG Split Block -----------------
class VGG_BW(nn.Module):
    def __init__(self, features, cfg, dataset, num_class,
                 init_weights=True):
        super(VGG_BW, self).__init__()

        self.features = features

        classifier = []
        if 'CIFAR' in dataset:
            classifier += [
                nn.Linear(cfg[-2], 512),
                # nn.BatchNorm1d(512),
                nn.Linear(512, num_class)
            ]
        else:
            raise RuntimeError("Not expected data flag !!!")

        self.classifier = nn.Sequential(*classifier)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def vgg_bw(cfg, block_wise, dataset, num_class):

    if block_wise:
        blocks = split_block(cfg)
        features = make_layers_block_wise(blocks, block_wise)
    else:
        features = make_layers_block_wise(cfg, block_wise)

    model = VGG_BW(features, cfg, dataset, num_class)

    return model


# ----------------- VGG Block -----------------
class VGG_Block(nn.Module):
    def __init__(self, features, cfg, dataset, num_class,
                 init_weights=True):
        super(VGG_Block, self).__init__()

        self.features = features

        if self.features:
            self.features = features
        else:
            classifier = []
            if 'CIFAR' in dataset:
                classifier += [
                    nn.Linear(cfg[-2], 512),
                    nn.BatchNorm1d(512),
                    nn.Linear(512, num_class)
                ]
            else:
                raise RuntimeError("Not expected data flag !!!")

            self.classifier = nn.Sequential(*classifier)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        if self.features:
            x = self.features(x)
        else:
            x = torch.flatten(x, 1)
            x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def vgg_block(cfg, block_id, dataset, num_class):
    blocks = split_block(cfg)
    if block_id > 0 and block_id < len(blocks):
        features = make_layers_block_wise(
            blocks[block_id], block_wise=False,
            in_channels=blocks[block_id-1][-2])
    elif block_id == 0:
        features = make_layers_block_wise(
            blocks[block_id], block_wise=False)
    else:
        features = None

    model = VGG_Block(features, cfg, dataset, num_class)

    return model



# ----------------- VGG Stock -----------------
class VGG_Stock(nn.Module):
    def __init__(self, features, cfg, dataset, num_class,
                 init_weights=True):
        super(VGG_Stock, self).__init__()

        self.features = features

        classifier = []
        if 'CIFAR' in dataset:
            classifier += [
                nn.Linear(cfg[-2], 512),
                nn.BatchNorm1d(512),
                nn.Linear(512, num_class)
            ]
        else:
            raise RuntimeError("Not expected data flag !!!")

        self.classifier = nn.Sequential(*classifier)

        self.scion_block = None
        self.scion_block_pos = -1
        self.scicon_len = 0

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # insert to features
        if self.scion_block is None:
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
        elif self.scion_block_pos > 0 and \
                self.scion_block_pos + self.scicon_len < len(self.features):
            x = self.features[:self.scion_block_pos](x)

            x = self.scion_block(x)

            x = self.features[self.scion_block_pos
                              +self.scicon_len:](x)

            x = torch.flatten(x, 1)
            x = self.classifier(x)
        elif self.scion_block_pos + self.scicon_len == len(self.features):
            x = self.features[:self.scion_block_pos](x)

            x = self.scion_block(x)

            x = torch.flatten(x, 1)
            x = self.classifier(x)
        # insert to and features and classifier
        elif self.scion_block_pos + self.scicon_len - 1 == len(self.features):
            x = self.features[:self.scion_block_pos](x)
            x = self.scion_block(x)

        elif self.scion_block_pos == 0:
            x = self.scion_block(x)

            x = self.features[self.scicon_len:](x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
        elif self.scion_block_pos == len(self.features):

            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.scion_block(x)
        else:
            raise RuntimeError("Out of range: {}/{}!".
                               format(self.scion_block_pos,
                                      len(self.features)))

        return x

    def set_scion(self, block, position, scicon_len):
        if scicon_len <= 0:
            raise RuntimeError("Illegal scicon length: {}".
                               format(scicon_len))
        if position + scicon_len - 1 > len(self.features):
            raise RuntimeError("Out of range: {}/{}!".
                               format(position,
                                      len(self.features)))
        self.scion_block = block
        self.scion_block_pos = position
        self.scicon_len = scicon_len

    def reset_scion(self):
        self.scion_block = None
        self.scion_block_pos = -1
        self.scicon_len = 0

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def vgg_stock(cfg, dataset, num_class):

    blocks = split_block(cfg)
    features = make_layers_block_wise(blocks, block_wise=True)

    model = VGG_Stock(features, cfg, dataset, num_class)

    return model


