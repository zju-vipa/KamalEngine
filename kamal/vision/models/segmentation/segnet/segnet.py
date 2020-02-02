from .layer import SegnetDown, SegnetUp
import torch.nn as nn 
from ...classification import vgg

__all__=[   'SegNet',
            'segnet_vgg11', 'segnet_vgg13','segnet_vgg16','segnet_vgg19',
            'segnet_vgg11_bn','segnet_vgg13_bn','segnet_vgg16_bn','segnet_vgg19_bn' ]

model_urls = {
    'segnet_vgg11': None,
    'segnet_vgg13': None,
    'segnet_vgg16': None,
    'segnet_vgg19': None,
    'segnet_vgg11_bn': None,
    'segnet_vgg13_bn': None,
    'segnet_vgg16_bn': None,
    'segnet_vgg19_bn': None,
}

_arch_dict = {
    'segnet_vgg11_bn': [1, 1, 2, 2, 2],
    'segnet_vgg13_bn': [2, 2, 2, 2, 2],
    'segnet_vgg16_bn': [2, 2, 3, 3, 3],
    'segnet_vgg19_bn': [2, 2, 4, 4, 4],
    'segnet_vgg11': [1, 1, 2, 2, 2],
    'segnet_vgg13': [2, 2, 2, 2, 2],
    'segnet_vgg16': [2, 2, 3, 3, 3],
    'segnet_vgg19': [2, 2, 4, 4, 4],
}

_backbone_dict = {
    'segnet_vgg11_bn': vgg.vgg11_bn,
    'segnet_vgg13_bn': vgg.vgg13_bn,
    'segnet_vgg16_bn': vgg.vgg16_bn,
    'segnet_vgg19_bn': vgg.vgg19_bn,
    'segnet_vgg11': vgg.vgg11,
    'segnet_vgg13': vgg.vgg13,
    'segnet_vgg16': vgg.vgg16,
    'segnet_vgg19': vgg.vgg19,
}


class SegNet( nn.Module ):
    def __init__(self, arch='segnet_vgg16_bn', num_classes=21, in_channels=3, pretrained_backbone=False, batch_norm=True, channel_list=(64, 128, 256, 512, 512)):
        super( SegNet, self ).__init__()
        assert len(channel_list)==5, 'length of channel_list must be 5'

        # predefined arch
        if isinstance(arch, str):
            arch_name = arch
            assert arch_name in _arch_dict.keys(), "arch_name for SegNet should be one of %s"%( _arch_dict.keys() )
            arch = _arch_dict[arch_name]
            batch_norm=True if 'bn' in arch_name else False
         # customized arch
        elif isinstance( arch, (list, tuple) ):
            arch_name = 'customized'

        self.num_classes = num_classes
        self.in_channels = in_channels
        
        self.down1 = SegnetDown(self.in_channels, channel_list[0], num_convs=arch[0], batch_norm=batch_norm) # 64
        self.down2 = SegnetDown(channel_list[0],  channel_list[1], num_convs=arch[1], batch_norm=batch_norm) # 128
        self.down3 = SegnetDown(channel_list[1],  channel_list[2], num_convs=arch[2], batch_norm=batch_norm) # 256
        self.down4 = SegnetDown(channel_list[2],  channel_list[3], num_convs=arch[3], batch_norm=batch_norm) # 512
        self.down5 = SegnetDown(channel_list[3],  channel_list[4], num_convs=arch[4], batch_norm=batch_norm) # 512

        self.up5 = SegnetUp(channel_list[4], channel_list[3], num_convs=arch[4], batch_norm=batch_norm)     # 512
        self.up4 = SegnetUp(channel_list[3], channel_list[2], num_convs=arch[3], batch_norm=batch_norm)     # 256
        self.up3 = SegnetUp(channel_list[2], channel_list[1], num_convs=arch[2], batch_norm=batch_norm)     # 128
        self.up2 = SegnetUp(channel_list[1], channel_list[0], num_convs=arch[1], batch_norm=batch_norm)     # 64
        self.up1 = SegnetUp(channel_list[0], self.num_classes, num_convs=arch[0], outer_most=True, batch_norm=batch_norm)

        if pretrained_backbone:
            assert arch_name!='customized', 'Only predefined archs have pretrained weights'
            self.load_from_pretrained_vgg(arch_name)
            
    def load_from_pretrained_vgg(self, vgg):
        if isinstance(vgg, str):
            vgg = _backbone_dict[ vgg ](pretrained=True)

        _blocks = [self.down1, self.down2, self.down3, self.down4, self.down5]
        segnet_features = []
        for _block in _blocks:
            for _layer in _block.layers:
                segnet_features.extend( _layer )

        vgg_features = [ layer for layer in vgg.features if not isinstance( layer, nn.MaxPool2d ) ] 

        for segnet_layer, vgg_layer in zip( segnet_features, vgg_features ):
            assert type( segnet_layer ) == type( vgg_layer ), "Inconsistant layer: %s, %s"%(type( segnet_features ), type( vgg_layer ))
            if isinstance( segnet_layer, nn.Conv2d ):
                segnet_layer.weight.data = vgg_layer.weight.data
                segnet_layer.bias.data = vgg_layer.bias.data
            elif isinstance( segnet_layer, nn.BatchNorm2d):
                segnet_layer.weight.data = vgg_layer.weight.data
                segnet_layer.bias.data = vgg_layer.bias.data
                segnet_layer.running_mean.data = vgg_layer.running_mean.data
                segnet_layer.running_var.data = vgg_layer.running_var.data
            
    def forward(self, inputs):
        down1, indices_1, unpool_shape1 = self.down1(inputs)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        down3, indices_3, unpool_shape3 = self.down3(down2)
        down4, indices_4, unpool_shape4 = self.down4(down3)
        down5, indices_5, unpool_shape5 = self.down5(down4)

        up5 = self.up5(down5, indices_5, unpool_shape5)
        up4 = self.up4(up5, indices_4, unpool_shape4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        up1 = self.up1(up2, indices_1, unpool_shape1)
        return up1

def segnet_vgg11(pretrained=False, progress=True, **kwargs):
    """Constructs a DeepLabV3+ model with a mobilenet backbone.
    """
    model = SegNet(arch='segnet_vgg11', **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

def segnet_vgg11_bn(pretrained=False, progress=True, **kwargs):
    """Constructs a DeepLabV3+ model with a mobilenet backbone.
    """
    model = SegNet(arch='segnet_vgg11_bn', **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

def segnet_vgg13(pretrained=False, progress=True, **kwargs):
    """Constructs a DeepLabV3+ model with a mobilenet backbone.
    """
    model = SegNet(arch='segnet_vgg13', **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

def segnet_vgg13_bn(pretrained=False, progress=True, **kwargs):
    """Constructs a DeepLabV3+ model with a mobilenet backbone.
    """
    model = SegNet(arch='segnet_vgg13_bn', **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def segnet_vgg16(pretrained=False, progress=True, **kwargs):
    """Constructs a DeepLabV3+ model with a mobilenet backbone.
    """
    model = SegNet(arch='segnet_vgg16', **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

def segnet_vgg16_bn(pretrained=False, progress=True, **kwargs):
    """Constructs a DeepLabV3+ model with a mobilenet backbone.
    """
    model = SegNet(arch='segnet_vgg16_bn', **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

def segnet_vgg19(pretrained=False, progress=True, **kwargs):
    """Constructs a DeepLabV3+ model with a mobilenet backbone.
    """
    model = SegNet(arch='segnet_vgg19', **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

def segnet_vgg19_bn(pretrained=False, progress=True, **kwargs):
    """Constructs a DeepLabV3+ model with a mobilenet backbone.
    """
    model = SegNet(arch='segnet_vgg19_bn', **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

if __name__=='__main__':
    import torch
    model = SegNet(num_classes=21)
    print(model)
    print( model( torch.randn(1,3,256,256) ).shape )
