import torch.nn as nn
from ...classification.resnet import BasicBlock, Bottleneck, resnet18, resnet34, resnet50, resnet101, resnet152

__all__= ['LinkNet', 'linknet_resnet18','linknet_resnet34','linknet_resnet50','linknet_resnet101','linknet_resnet152']

model_urls = {
    'linknet_resnet18': None,
    'linknet_resnet34': None,
    'linknet_resnet50': None,
    'linknet_resnet101': None,
    'linknet_resnet152': None,
}

_arch_dict = {
    'linknet_resnet18':  ( (2, 2, 2, 2), BasicBlock ),
    'linknet_resnet34':  ( (2, 2, 2, 2),   BasicBlock ),
    'linknet_resnet50':  ( (3, 4, 6, 3),   Bottleneck ),
    'linknet_resnet101': ( (3, 4, 23, 3),  Bottleneck ),
    'linknet_resnet152': ( (3, 8, 36, 3),  Bottleneck ),
}

_backbone_dict = {
    'linknet_resnet18':  resnet18,
    'linknet_resnet34':  resnet34,
    'linknet_resnet50':  resnet50,
    'linknet_resnet101': resnet101,
    'linknet_resnet152': resnet152,
}

__all__=['LinkNet']

class LinkNetDecoder(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1):
        super(LinkNetDecoder, self).__init__( 
            nn.Conv2d(in_channels, in_channels//4, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(inplace=True),

            # upsample
            nn.ConvTranspose2d(in_channels//4, in_channels//4, kernel_size=3, stride=stride, padding=1, output_padding=int(stride==2)),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels//4, out_channels, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

class LinkNet(nn.Module):
    def __init__(self, arch='linknet_resnet18', num_classes=21, in_channels=3, pretrained_backbone=False, channel_list=(64, 128, 256, 512 ), block=BasicBlock):
        super(LinkNet, self).__init__()

        # predefined arch
        if isinstance(arch, str):
            arch_name = arch
            assert arch_name in _arch_dict.keys(), "arch_name for SegNet should be one of %s"%( _arch_dict.keys() )
            arch, block = _arch_dict[arch_name]
         # customized arch
        elif isinstance( arch, (list, tuple) ):
            arch_name = 'customized'

        # Encoder
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inplanes = 64

        # Encoder
        self.layer1 = self._make_layer(block, planes=channel_list[0], blocks=arch[0])
        self.layer2 = self._make_layer(block, planes=channel_list[1], blocks=arch[1], stride=2)
        self.layer3 = self._make_layer(block, planes=channel_list[2], blocks=arch[2], stride=2)
        self.layer4 = self._make_layer(block, planes=channel_list[3], blocks=arch[3], stride=2)

        decoder_channel_list = [ c*block.expansion for c in channel_list ]
        # Decoder
        self.decoder4 = LinkNetDecoder(decoder_channel_list[3], decoder_channel_list[2], stride=2)
        self.decoder3 = LinkNetDecoder(decoder_channel_list[2], decoder_channel_list[1], stride=2)
        self.decoder2 = LinkNetDecoder(decoder_channel_list[1], decoder_channel_list[0], stride=2)
        self.decoder1 = LinkNetDecoder(decoder_channel_list[0], decoder_channel_list[0])

        # Final Classifier
        self.classifier = nn.Sequential(
            nn.ConvTranspose2d(decoder_channel_list[0], 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, num_classes, kernel_size=2, stride=2, padding=0)
        )

        if pretrained_backbone:
            self.load_from_pretrained_resnet(arch_name)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def load_from_pretrained_resnet(self, resnet):
        if isinstance(resnet, str):
            resnet = _backbone_dict[ resnet ](pretrained=True)

        def copy_params(layer1, layer2):
            for p1, p2 in zip( layer1.parameters(), layer2.parameters() ):
                p1.data = p2.data

        linknet_part = [ self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4 ]
        resnet_part = [ resnet.conv1, resnet.bn1, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4 ]

        for linknet_layer, resnet_layer in zip( linknet_part, resnet_part ):
            copy_params( linknet_layer, resnet_layer )
                
    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        e1 = self.layer1(x)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)

        d4 = self.decoder4(e4)
        d4 = d4+e3
        d3 = self.decoder3(d4)
        d3 = d3+e2
        d2 = self.decoder2(d3)
        d2 = d2+e1
        d1 = self.decoder1(d2)

        return self.classifier(d1)


def linknet_resnet18(pretrained=False, progress=True, **kwargs):
    model = LinkNet(arch='linknet_resnet18', **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

def linknet_resnet34(pretrained=False, progress=True, **kwargs):
    model = LinkNet(arch='linknet_resnet34', **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

def linknet_resnet50(pretrained=False, progress=True, **kwargs):
    model = LinkNet(arch='linknet_resnet50', **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

def linknet_resnet101(pretrained=False, progress=True, **kwargs):
    model = LinkNet(arch='linknet_resnet101', **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

def linknet_resnet152(pretrained=False, progress=True, **kwargs):
    model = LinkNet(arch='linknet_resnet152', **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

