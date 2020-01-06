from torchvision.models import vgg11
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.num_classes = num_classes
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
        )

        if init_weights:
            self._initialize_weights()

    def adjust(self):
        blocks = nn.ModuleList()
        layers = []
        skip = True
        for l in self.features:
            layers.append(l)
            if isinstance(l, nn.MaxPool2d):
                if skip:
                    skip = False
                    continue
                blocks.append(nn.Sequential(*layers))
                layers = []
        self.features = blocks
        del self.classifier
        self.classifier = nn.Linear(512 * 7 * 7, self.num_classes)

    def forward(self, x):
        block_out = []
        for b in self.features:
            x = b(x)
            block_out.append(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x, block_out

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


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A']), **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['vgg11'])
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if 'classifier' not in k}
        model.load_state_dict(pretrained_dict)
    return model


def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['vgg11_bn'])
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if 'classifier' not in k}
        model.load_state_dict(pretrained_dict)

        # model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    model.adjust()
    return model


def vgg13(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B']), **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['vgg13'])
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if 'classifier' not in k}
        model.load_state_dict(pretrained_dict)

        # model.load_state_dict(model_zoo.load_url(model_urls['vgg13']))
    model.adjust()
    return model


def vgg13_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['vgg13_bn'])
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if 'classifier' not in k}
        model.load_state_dict(pretrained_dict)
        # model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']))
    model.adjust()
    return model


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['vgg16'])
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if 'classifier' not in k}
        model.load_state_dict(pretrained_dict)
        # model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    model.adjust()
    return model


def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['vgg16_bn'])
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if 'classifier' not in k}
        model.load_state_dict(pretrained_dict)
        # model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    model.adjust()
    return model


def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['vgg19'])
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if 'classifier' not in k}
        model.load_state_dict(pretrained_dict)
        # model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    model.adjust()
    return model


def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['vgg19_bn'])
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if 'classifier' not in k}
        model.load_state_dict(pretrained_dict)
        # model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    model.adjust()
    return model
