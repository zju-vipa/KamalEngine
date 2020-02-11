import torch.nn as nn
import torch.nn.functional as F
import torch 
from .layer import DoubleConv, Down, Up

__all__=['UNet', 'unet']

model_urls = {
    'unet': None,
}

class UNet(nn.Module):
    def __init__(self, num_classes=21, in_channels=3, deconv=True, batch_norm=True, channel_list=(64, 128, 256, 512, 1024)):
        super(UNet, self).__init__()
        assert len(channel_list)==5, 'length of channel_list must be 5'
        # downsampling
        self.down1 = Down(in_channels,      channel_list[0], batch_norm)        # 64
        self.down2 = Down(channel_list[0],  channel_list[1], batch_norm)        # 128
        self.down3 = Down(channel_list[1],  channel_list[2], batch_norm)        # 256
        self.down4 = Down(channel_list[2],  channel_list[3], batch_norm)        # 512
        self.center = DoubleConv(channel_list[3], channel_list[4], batch_norm)  # 1024

        # upsampling
        self.up4 = Up(channel_list[4],  channel_list[3], batch_norm, deconv)    # 512
        self.up3 = Up(channel_list[3],  channel_list[2], batch_norm, deconv)    # 256
        self.up2 = Up(channel_list[2],  channel_list[1], batch_norm, deconv)    # 128
        self.up1 = Up(channel_list[1],  channel_list[0],  batch_norm, deconv)   # 64

        self.classifier = nn.Conv2d(channel_list[0], num_classes, 1)

    def forward(self, inputs):
        out_size = inputs.shape[2:]
        out, conv_features1 = self.down1(inputs)
        out, conv_features2 = self.down2(out)
        out, conv_features3 = self.down3(out)
        out, conv_features4 = self.down4(out)

        out = self.center(out)

        out = self.up4(out, conv_features4)
        out = self.up3(out, conv_features3)
        out = self.up2(out, conv_features2)
        out = self.up1(out, conv_features1)

        out = self.classifier(out)
        if out.shape[2:]!=out_size:
            out = nn.functional.interpolate( out, size=out_size, mode='bilinear', align_corners=True )
        return out

def unet(pretrained=False, progress=True, **kwargs):
    """Constructs a DeepLabV3+ model with a mobilenet backbone.
    """
    model = UNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model