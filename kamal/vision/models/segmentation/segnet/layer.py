# Modified from https://github.com/meetshah1995/pytorch-semseg/blob/801fb20054/ptsemseg/models/segnet.py
import torch.nn as nn

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

class SegnetDown(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs=2, batch_norm=True):
        super(SegnetDown, self).__init__()
        layers = [ConvBNRelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, batch_norm=batch_norm)]
        for _ in range(num_convs-1):
            layers.append( ConvBNRelu(out_channels, out_channels, kernel_size=3, stride=1, padding=1, batch_norm=batch_norm) )

        self.layers = nn.Sequential( *layers ) 
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.layers(inputs)
        ori_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, ori_shape

class SegnetUp(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs=2, outer_most=False, batch_norm=True):
        super(SegnetUp, self).__init__()
        if outer_most:
            batch_norm = False
            activation = False
        else:
            activation = True

        layers = []
        for _ in range(num_convs-1):
            layers.append( ConvBNRelu(in_channels, in_channels, kernel_size=3, stride=1, padding=1, batch_norm=batch_norm) )
        # remove relu if it is the outer most layer
        layers.append( ConvBNRelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, batch_norm=batch_norm, activation=activation) )
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.layers = nn.Sequential( *layers ) 
        
    def forward(self, inputs, indices, ori_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=ori_shape)
        outputs = self.layers(outputs)
        return outputs

