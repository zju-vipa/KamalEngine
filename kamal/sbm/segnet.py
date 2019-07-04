#coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SegNet(nn.Module):
    @staticmethod
    def _get_block_features(in_channels, out_channels, repeated_times, is_encoder):
        assert(repeated_times >=1, 'SegNet block中 至少有一个layer')
        if is_encoder:
            layer_list = [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels), # TODO: bn?
                    nn.ReLU(inplace=True)                
                ] + [
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)                
                ] * (repeated_times-1)
        else:
            layer_list = [
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(in_channels), # TODO: bn?
                    nn.ReLU(inplace=True)                
                ] * (repeated_times-1) + [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)                
                ]
        return nn.Sequential(*layer_list)

    def __init__(self, in_channels, num_classes, channels = 1024):
        super(SegNet, self).__init__()
        self.enc1 = SegNet._get_block_features(in_channels, channels / 16, 2, True)
        self.enc2 = SegNet._get_block_features(channels / 16, channels / 8, 2, True)
        self.enc3 = SegNet._get_block_features(channels / 8, channels / 4, 3, True)
        self.enc4 = SegNet._get_block_features(channels / 4, channels / 2, 3, True)
        self.enc5 = SegNet._get_block_features(channels / 2, channels / 2, 3, True)

        self.dec5 = SegNet._get_block_features(channels / 2, channels / 2, 3, False)
        self.dec4 = SegNet._get_block_features(channels / 2, channels / 4, 3, False)
        self.dec3 = SegNet._get_block_features(channels / 4, channels / 8, 3, False)
        self.dec2 = SegNet._get_block_features(channels / 8, channels / 16, 2, False)
        self.dec1 = SegNet._get_block_features(channels / 16, channels / 16, 2, False)

        self.classifier = nn.Sequential(
            nn.Conv2d(channels / 16, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # encoder
        x1 = self.enc1(x)
        e1, m1 = F.max_pool2d(x1, kernel_size=2, stride=2, return_indices=True)
        x2 = self.enc2(e1)
        e2, m2 = F.max_pool2d(x2, kernel_size=2, stride=2, return_indices=True)
        x3 = self.enc3(e2)
        e3, m3 = F.max_pool2d(x3, kernel_size=2, stride=2, return_indices=True)
        x4 = self.enc4(e3)
        e4, m4 = F.max_pool2d(x4, kernel_size=2, stride=2, return_indices=True)
        x5 = self.enc5(e4)
        e5, m5 = F.max_pool2d(x5, kernel_size=2, stride=2, return_indices=True)

        # print(x1.size(), x2.size(), x3.size(), x4.size(), x5.size())
        # decoder
        d5 = self.dec5(F.max_unpool2d(e5, m5, kernel_size=2, stride=2, output_size=x5.size()))
        d4 = self.dec4(F.max_unpool2d(d5, m4, kernel_size=2, stride=2, output_size=x4.size()))
        d3 = self.dec3(F.max_unpool2d(d4, m3, kernel_size=2, stride=2, output_size=x3.size()))
        d2 = self.dec2(F.max_unpool2d(d3, m2, kernel_size=2, stride=2, output_size=x2.size()))
        d1 = self.dec1(F.max_unpool2d(d2, m1, kernel_size=2, stride=2, output_size=x1.size()))
        return self.classifier(d1)


