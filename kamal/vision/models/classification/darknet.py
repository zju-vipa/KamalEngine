import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from ..utils import download_from_url, load_darknet_weights

model_urls = {
    'darknet19':     'https://pjreddie.com/media/files/darknet19.weights',
    'darknet19_448': 'https://pjreddie.com/media/files/darknet19_448.weights',
    'darknet53':     'https://pjreddie.com/media/files/darknet53.weights',
    'darknet53_448': 'https://pjreddie.com/media/files/darknet53_448.weights'
}

def conv3x3(in_planes, out_planes, padding=1, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, planes, norm_layer=None, residual=True):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.residual = residual
        self.block = nn.Sequential(
            conv1x1(planes, planes//2),
            norm_layer(planes//2),
            nn.LeakyReLU(0.1, inplace=True),

            conv3x3(planes//2, planes),
            norm_layer(planes),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        identity = x
        out = self.block(x)
        if self.residual:
            out = out+identity
        return out

class DarkNet( nn.Module ):
    def __init__(self, layers, num_classes=1000, pooling=False, residual=True):
        super(DarkNet, self).__init__()
        self.inplanes = 32
        self.pooling = pooling
        self.residual = residual
        
        features = [
            conv3x3(3, self.inplanes),
            nn.BatchNorm2d(self.inplanes),
            nn.LeakyReLU(0.1, inplace=True),
        ]
        features.extend( self._make_layer(64,   layers[0]) )
        features.extend( self._make_layer(128,  layers[1]) )
        features.extend( self._make_layer(256,  layers[2]) )
        features.extend( self._make_layer(512,  layers[3]) )
        features.extend( self._make_layer(1024, layers[4]) )

        self.features = nn.Sequential( *features )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1024, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks):
        layers = []
        if self.pooling==True: 
            layers.append( nn.MaxPool2d(2,2) ) # downsample with maxpooling
        layers.extend([
            conv3x3(self.inplanes, planes, stride=1 if self.pooling else 2),
            nn.BatchNorm2d(planes),
            nn.LeakyReLU(0.1, inplace=True),
        ])
        
        for _ in range(blocks):
            layers.append(BasicBlock(planes, residual=self.residual))
        self.inplanes = planes
        return layers

    def load_weights(self, weights_file):
        load_darknet_weights( self, weights_file )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
                    
def darknet19(pretrained=False, progress=True, **kwargs):
    model = DarkNet(layers=[ 0, 1, 1, 2, 2 ], pooling=True, residual=False)
    if pretrained:
        weights_file = download_from_url(model_urls['darknet19'], progress=progress)
        load_darknet_weights( model, weights_file )
    return model

def darknet19_448(pretrained=False, progress=True, **kwargs):
    model = DarkNet(layers=[ 0, 1, 1, 2, 2 ], pooling=True, residual=False)
    if pretrained:
        weights_file = download_from_url(model_urls['darknet19_448'], progress=progress)
        load_darknet_weights( model, weights_file )
    return model

def darknet53(pretrained=False, progress=True, **kwargs):
    model = DarkNet(layers=[ 1, 2, 8, 8, 4 ], pooling=False, residual=True)
    if pretrained:
        weights_file = download_from_url(model_urls['darknet53'], progress=progress)
        load_darknet_weights( model, weights_file )
    return model

def darknet53_448(pretrained=False, progress=True, **kwargs):
    model = DarkNet(layers=[ 1, 2, 8, 8, 4 ], pooling=False, residual=True)
    if pretrained:
        weights_file = download_from_url(model_urls['darknet53_448'], progress=progress)
        load_darknet_weights( model, weights_file )
    return model