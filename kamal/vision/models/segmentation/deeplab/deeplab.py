from .utils import IntermediateLayerGetter
from .layer import DeepLabv3Head, DeepLabv3PlusHead
from ...classification import mobilenetv2, resnet

import torch.nn as nn 
import torch.nn.functional as F

from torchvision.models.utils import load_state_dict_from_url

__all__=['DeepLabV3', 
        'deeplabv3_mobilenetv2', 'deeplabv3_resnet50', 'deeplabv3_resnet101',
        'deeplabv3plus_mobilenetv2', 'deeplabv3plus_resnet50', 'deeplabv3plus_resnet101']

model_urls = {
    'deeplabv3_mobilenetv2': None,
    'deeplabv3_resnet50': None,
    'deeplabv3_resnet101': None,

    'deeplabv3plus_mobilenetv2': None,
    'deeplabv3plus_resnet50': None,
    'deeplabv3plus_resnet101': None,
}

class DeepLabV3(nn.Module):
    def __init__(self, arch='deeplabv3_mobilenetv2', num_classes=21, output_stride=8, pretrained_backbone=False):
        super(DeepLabV3, self).__init__()
        assert arch in __all__[1:], "arch_name for deeplab should be one of %s"%( __all__[1:] )

        arch_type, backbone_name = arch.split('_')

        if backbone_name=='mobilenetv2':
            backbone, classifier = _segm_mobilenet(arch_type, backbone_name, num_classes, 
                output_stride=output_stride, pretrained_backbone=pretrained_backbone)
        elif backbone_name.startswith('resnet'):
            backbone, classifier = _segm_resnet(arch_type, backbone_name, num_classes, 
                output_stride=output_stride, pretrained_backbone=pretrained_backbone)
        else:
            print("backbone nam")
            raise NotImplementedError

        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x

def _segm_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone):

    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)
    
    inplanes = 2048
    low_level_planes = 256

    if name=='deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabv3PlusHead(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'layer4': 'out'}
        classifier = DeepLabv3Head(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    #model = DeepLabV3(backbone, classifier)
    return backbone, classifier

def _segm_mobilenet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride==8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    backbone = mobilenetv2.mobilenet_v2(pretrained=pretrained_backbone, output_stride=output_stride)
    
    backbone.low_level_features = backbone.features[0:4]
    backbone.high_level_features = backbone.features[4:-1]
    backbone.features = None
    backbone.classifier = None

    inplanes = 320
    low_level_planes = 24
    
    if name=='deeplabv3plus':
        return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
        classifier = DeepLabv3PlusHead(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'high_level_features': 'out'}
        classifier = DeepLabv3Head(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    #model = DeepLabV3(backbone, classifier)
    return backbone, classifier

def deeplabv3_mobilenetv2(pretrained=False, progress=True, **kwargs):
    model = DeepLabV3(arch='deeplabv3_mobilenetv2', **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

def deeplabv3_resnet50(pretrained=False, progress=True, **kwargs):
    model = DeepLabV3(arch='deeplabv3_resnet50', **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

def deeplabv3_resnet101(pretrained=False, progress=True, **kwargs):
    model = DeepLabV3(arch='deeplabv3_resnet101', **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def deeplabv3plus_mobilenetv2(pretrained=False, progress=True, **kwargs):
    model = DeepLabV3(arch='deeplabv3plus_mobilenetv2', **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

def deeplabv3plus_resnet50(pretrained=False, progress=True, **kwargs):
    model = DeepLabV3(arch='deeplabv3plus_resnet50', **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

def deeplabv3plus_resnet101(pretrained=False, progress=True, **kwargs):
    model = DeepLabV3(arch='deeplabv3plus_resnet101', **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model