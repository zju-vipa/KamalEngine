import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ResBlock(nn.Module):
    """ Residual Blocks
    """
    def __init__(self, inplanes, planes, stride=1, momentum=0.1):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        #self.bn1 = nn.BatchNorm2d(planes, momentum=momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        #self.bn2 = nn.BatchNorm2d(planes, momentum=momentum)
        if stride > 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1,
                          stride=stride, bias=False),
                #nn.BatchNorm2d(planes, momentum=momentum)
            )
        else:
            self.downsample = None

        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        #out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class CFL_ConvBlock(nn.Module):
    """Common Feature Blocks for Convolutional layer
    
    This module is used to capture the common features of multiple teachers and calculate mmd with features of student.

    **Parameters:**
        - **channel_s** (int): channel of student features
        - **channel_t** (list or tuple): channel list of teacher features
    """
    def __init__(self, channel_s, channel_t, channel_h, k_size=5):
        super(CFL_ConvBlock, self).__init__()

        self.align_t = nn.ModuleList()
        for ch_t in channel_t:
            self.align_t.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=ch_t, out_channels=2*channel_h,
                              kernel_size=1, bias=False),
                    nn.ReLU(inplace=True)
                )
            )

        self.align_s = nn.Sequential(
            nn.Conv2d(in_channels=channel_s, out_channels=2*channel_h,
                      kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.extractor = nn.Sequential(
            ResBlock(inplanes=2*channel_h, planes=channel_h, stride=1),
            ResBlock(inplanes=channel_h, planes=channel_h, stride=1),
            ResBlock(inplanes=channel_h, planes=channel_h, stride=1),
        )

        self.dec_t = nn.ModuleList()
        for ch_t in channel_t:
            self.dec_t.append(
                nn.Sequential(
                    nn.Conv2d(channel_h, ch_t, kernel_size=3,
                              stride=1, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ch_t, ch_t, kernel_size=1,
                              stride=1, padding=0, bias=False)
                )
            )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, fs, ft):
        aligned_t = [self.align_t[i](ft[i]) for i in range(len(ft))]
        aligned_s = self.align_s(fs)

        ht = [self.extractor(f) for f in aligned_t]
        hs = self.extractor(aligned_s)

        ft_ = [self.dec_t[i](ht[i]) for i in range(len(ht))]
        return (hs, ht), (ft_, ft)
