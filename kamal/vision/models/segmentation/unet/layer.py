# Modified from https://github.com/meetshah1995/pytorch-semseg
import torch.nn as nn
import torch.nn.functional as F
import torch 

class ConvBNRelu(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, dilation=1, batch_norm=True, activation=True):
        conv_mod = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=kernel_size, padding=padding, stride=stride, bias=bias, dilation=dilation)
        if batch_norm:
            cbr_unit = [conv_mod, nn.BatchNorm2d(int(out_channels)) ]
        else:
            cbr_unit = [ conv_mod ]

        if activation==True:
            cbr_unit.append( nn.ReLU(inplace=True) )
        super(ConvBNRelu, self).__init__(*cbr_unit)

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, batch_norm):
        super(DoubleConv, self).__init__(
            ConvBNRelu( in_channels,  out_channels, 3, 1, 1, batch_norm=batch_norm),
            ConvBNRelu( out_channels, out_channels, 3, 1, 1, batch_norm=batch_norm)
        )
    
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(Down, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels, batch_norm )
        self.downsample = nn.MaxPool2d(2,2)

    def forward(self, inputs):
        conv_features = self.double_conv(inputs)
        outputs = self.downsample(conv_features)
        return outputs, conv_features

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, deconv=True):
        super(Up, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2) if deconv else nn.UpsamplingBilinear2d(scale_factor=2)
        self.double_conv = DoubleConv(in_channels, out_channels, batch_norm ) 

    def forward(self, inputs, skip_inputs):
        outputs = self.upsample(inputs)
        padding_h = skip_inputs.size()[2] - outputs.size()[2]
        padding_w = skip_inputs.size()[3] - outputs.size()[3]
        padding = [ padding_w//2, padding_w-padding_w//2, padding_h//2, padding_h-padding_h//2 ]
        outputs = F.pad(outputs, pad=padding)

        outputs = torch.cat( [ skip_inputs, outputs ], dim=1 )
        outputs = self.double_conv(outputs)
        return outputs