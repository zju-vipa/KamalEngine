# coding:utf-8
import os
import math
import numpy as mp
from .modules import torch_load

class InfoStruct:
    def __init__(self,typename, in_channels=None, out_channels=None,kernel_size=None,stride=1,padding=0 ):
        self.typename = typename
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def getargs(self):
        return {'in_channels': self.in_channels, 'out_channels': self.out_channels, 'kernel_size': self.kernel_size,
                'stride': self.stride, 'padding': self.padding}

    def getinfo(self):
        return self.typename, self.in_channels, self.out_channels, self.kernel_size


def parser(model):
    test = torch_load(model)
    namelist = []
    nameinfo = []
    modulelist = []
    count = 0
    for id_x, m in enumerate(test.modules()):
        if m.__class__.__name__ == "Sequential":
            for id_y, m1 in enumerate(m.modules()):
                if m1.__class__.__name__ != "Sequential":
                    namelist.append(m1.__class__.__name__)
                    count += 1

                    if m1.__class__.__name__ == "Conv2d":
                        tmp = InfoStruct(typename='Conv2d', in_channels=m1.in_channels, out_channels=m1.out_channels,
                                         kernel_size=m1.kernel_size, stride=m1.stride, padding=m1.padding)
                    elif m1.__class__.__name__ == "Linear":
                        tmp = InfoStruct(
                            typename='Linear', in_channels=m1.in_features, out_channels=m1.out_features)
                    else:
                        tmp = InfoStruct(typename=m1.__class__.__name__)

                    nameinfo.append(tmp)
                    modulelist.append(m1)
    return namelist, nameinfo, modulelist
