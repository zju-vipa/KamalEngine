import torch.nn as nn
import torch
import math

# ------------ Now there is only Resnet18 model ------------
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# BasicBlock without last relu
class BasicBlock_last(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_last, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out

class ResNet(nn.Module):

    def __init__(self, block, block_last, layers, num_classes=1000, channel_num=[64, 64, 128, 256, 512]):
        self.inplanes = channel_num[0]
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, channel_num[0], kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(channel_num[0])
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer_modified(block, block_last, channel_num[1], layers[0])
        self.relu1 = nn.ReLU(inplace=False)
        self.layer2 = self._make_layer_modified(block, block_last, channel_num[2], layers[1], stride=2)
        self.relu2 = nn.ReLU(inplace=False)
        self.layer3 = self._make_layer_modified(block, block_last, channel_num[3], layers[2], stride=2)
        self.relu3 = nn.ReLU(inplace=False)
        self.layer4 = self._make_layer_modified(block, block_last, channel_num[4], layers[3], stride=2)
        self.relu4 = nn.ReLU(inplace=False)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc_layer = nn.Linear(channel_num[4] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer_modified(self, block, block_last, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []

        #first basicblock
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        #middle basicblock
        for i in range(1, blocks-1):
            layers.append(block(self.inplanes, planes))

        #last basicblock
        layers.append(block_last(self.inplanes, planes))


        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.relu1(x)

        x = self.layer2(x)
        x = self.relu2(x)

        x = self.layer3(x)
        x = self.relu3(x)

        x = self.layer4(x)
        x = self.relu4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)

        return x

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, BasicBlock_last, [2, 2, 2, 2], **kwargs)

    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        # params = torch.load('models/resnet50-19c8e357.pth')
        pretrained_model = '/examples/amalgamation/customize_ka/weights/resnet/resnet18-5c106cde.pth'
        print('load pretrained model from: {}'.format(pretrained_model))
        params = torch.load(pretrained_model)
        del params['fc.weight']
        del params['fc.bias']
        model.load_state_dict(params, strict=False)
    return model

class ResNet_MultiTask(nn.Module):

    def __init__(self, block, block_last, channel_nums, layers, target_attributes, num_classes=1000):
        self.inplanes = channel_nums[0]
        super(ResNet_MultiTask, self).__init__()
        self.conv1 = nn.Conv2d(3, channel_nums[0], kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(channel_nums[0])
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU(inplace=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer_modified(block, block_last, channel_nums[1], layers[0])
        self.layer2 = self._make_layer_modified(block, block_last, channel_nums[2], layers[1], stride=2)
        self.layer3 = self._make_layer_modified(block, block_last, channel_nums[3], layers[2], stride=2)
        self.layer4 = self._make_layer_modified(block, block_last, channel_nums[4], layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.target_num = len(target_attributes)

        self.fc_layer = nn.ModuleList()

        for i in range(self.target_num):
            self.fc_layer.append(nn.Linear(channel_nums[4] * block.expansion, num_classes))


        # self.fc1 = nn.Linear(512 * block.expansion, num_classes)
        # self.fc2 = nn.Linear(512 * block.expansion, num_classes)
        # self.fc3 = nn.Linear(512 * block.expansion, num_classes)
        # self.fc4 = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer_modified(self, block, block_last, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []

        #first basicblock
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        #middle basicblock
        for i in range(1, blocks-1):
            layers.append(block(self.inplanes, planes))

        #last basicblock
        layers.append(block_last(self.inplanes, planes))


        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = self.relu(x)

        x = self.layer3(x)
        x = self.relu(x)

        feature4 = self.layer4(x)
        x = self.relu(feature4)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        results = []

        for layer in self.fc_layer:
            results.append(layer(x))

        # x = self.fc(x)
        # t1 = self.fc1(x)
        # t2 = self.fc2(x)
        # t3 = self.fc3(x)
        # t4 = self.fc4(x)

        # return results, feature4
        return results

def resnet18_multitask(channel_nums, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_MultiTask(BasicBlock, BasicBlock_last, channel_nums, [2, 2, 2, 2], **kwargs)

    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        # params = torch.load('models/resnet50-19c8e357.pth')
        pretrained_model = './weights/resnet/resnet18-5c106cde.pth'
        print('load pretrained model from: {}'.format(pretrained_model))
        params = torch.load(pretrained_model)
        del params['fc.weight']
        del params['fc.bias']
        model.load_state_dict(params, strict=False)
    return model

class ResNet_c(nn.Module):

    def __init__(self, block, block_last, layers, num_classes=1000, channel_num=[64, 64, 128, 256, 512]):
        self.inplanes = channel_num[0]
        super(ResNet_c, self).__init__()
        self.conv1 = nn.Conv2d(3, channel_num[0], kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(channel_num[0])
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer_modified(block, block_last, channel_num[1], layers[0])
        self.relu1 = nn.ReLU(inplace=False)
        self.layer2 = self._make_layer_modified(block, block_last, channel_num[2], layers[1], stride=2)
        self.relu2 = nn.ReLU(inplace=False)
        self.layer3 = self._make_layer_modified(block, block_last, channel_num[3], layers[2], stride=2)
        self.relu3 = nn.ReLU(inplace=False)
        self.layer4 = self._make_layer_modified(block, block_last, channel_num[4], layers[3], stride=2)
        self.relu4 = nn.ReLU(inplace=False)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc_layer = nn.Linear(channel_num[4] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer_modified(self, block, block_last, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []

        #first basicblock
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        #middle basicblock
        for i in range(1, blocks-1):
            layers.append(block(self.inplanes, planes))

        #last basicblock
        layers.append(block_last(self.inplanes, planes))


        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.relu1(x)

        x = self.layer2(x)
        x = self.relu2(x)

        x = self.layer3(x)
        x = self.relu3(x)

        feature4 = self.layer4(x)

        x = self.relu4(feature4)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)

        # return x, feature4
        return x

class ResNet_t(nn.Module):

    def __init__(self, block, block_last, layers, target_attributes, num_classes=1000, channel_num=[72, 72, 72*2, 72*4, 72*8]):
        self.inplanes = channel_num[0]
        super(ResNet_t, self).__init__()
        self.conv1 = nn.Conv2d(3, channel_num[0], kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(channel_num[0])
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer_modified(block, block_last, channel_num[1], layers[0])
        self.relu1 = nn.ReLU(inplace=False)
        self.layer2 = self._make_layer_modified(block, block_last, channel_num[2], layers[1], stride=2)
        self.relu2 = nn.ReLU(inplace=False)
        self.layer3 = self._make_layer_modified(block, block_last, channel_num[3], layers[2], stride=2)
        self.relu3 = nn.ReLU(inplace=False)
        self.layer4 = self._make_layer_modified(block, block_last, channel_num[4], layers[3], stride=2)
        self.relu4 = nn.ReLU(inplace=False)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.target_num = len(target_attributes)

        self.fc_layer = nn.ModuleList()

        for i in range(self.target_num):
            self.fc_layer.append(nn.Linear(channel_num[0] * 8 * block.expansion, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer_modified(self, block, block_last, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []

        #first basicblock
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        #middle basicblock
        for i in range(1, blocks-1):
            layers.append(block(self.inplanes, planes))

        #last basicblock
        layers.append(block_last(self.inplanes, planes))


        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.relu1(x)

        x = self.layer2(x)
        x = self.relu2(x)

        x = self.layer3(x)
        x = self.relu3(x)

        feature4 = self.layer4(x)

        x = self.relu4(feature4)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        results = []

        for layer in self.fc_layer:
            results.append(layer(x))

        return results

class ResNet_s(nn.Module):

    def __init__(self, block, block_last, channel_nums, layers, target_attributes, num_classes=1000):
        self.inplanes = channel_nums[0]
        super(ResNet_s, self).__init__()
        self.conv1 = nn.Conv2d(3, channel_nums[0], kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(channel_nums[0])
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU(inplace=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer_modified(block, block_last, channel_nums[1], layers[0])
        self.layer2 = self._make_layer_modified(block, block_last, channel_nums[2], layers[1], stride=2)
        self.layer3 = self._make_layer_modified(block, block_last, channel_nums[3], layers[2], stride=2)
        self.layer4 = self._make_layer_modified(block, block_last, channel_nums[4], layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.target_num = len(target_attributes)

        self.fc_layer = nn.ModuleList()

        for i in range(self.target_num):
            self.fc_layer.append(nn.Linear(channel_nums[4] * block.expansion, num_classes))


        # self.fc1 = nn.Linear(512 * block.expansion, num_classes)
        # self.fc2 = nn.Linear(512 * block.expansion, num_classes)
        # self.fc3 = nn.Linear(512 * block.expansion, num_classes)
        # self.fc4 = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer_modified(self, block, block_last, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []

        #first basicblock
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        #middle basicblock
        for i in range(1, blocks-1):
            layers.append(block(self.inplanes, planes))

        #last basicblock
        layers.append(block_last(self.inplanes, planes))


        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = self.relu(x)

        x = self.layer3(x)
        x = self.relu(x)

        feature4 = self.layer4(x)
        x = self.relu(feature4)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        results = []

        for layer in self.fc_layer:
            results.append(layer(x))

        # x = self.fc(x)
        # t1 = self.fc1(x)
        # t2 = self.fc2(x)
        # t3 = self.fc3(x)
        # t4 = self.fc4(x)

        # return results, feature4
        return results

class EncoderModule(nn.Module):
    def __init__(self, input_channel, output_channel, m_no=None, is_m=False, s=False, init_weights=False):
        super(EncoderModule, self).__init__()
        self.is_s = s
        self.is_m = is_m

        self.conv0 = nn.Conv2d(input_channel[0], output_channel[0], kernel_size=1)
        self.conv1 = nn.Conv2d(input_channel[1], output_channel[1], kernel_size=1)
        self.conv2 = nn.Conv2d(input_channel[2], output_channel[2], kernel_size=1)
        self.conv3 = nn.Conv2d(input_channel[3], output_channel[3], kernel_size=1)
        self.conv4 = nn.Conv2d(input_channel[4], output_channel[4], kernel_size=1)
        if self.is_s:
            # self.m = nn.Parameter(torch.tensor(1.0, requires_grad=True))
            # self.m = nn.Parameter(torch.ones(1), requires_grad=True)
            self.m = nn.Parameter(torch.ones(1), requires_grad=True)
            # self.m = nn.Linear(1)
        if self.is_m:
            self.m_no = m_no
            # self.m = nn.Parameter(torch.tensor(1.0, requires_grad=True))
            # self.m = nn.Parameter(torch.ones(1), requires_grad=True)
            self.m = nn.Parameter(torch.ones(1), requires_grad=True)
            # self.m = nn.Linear(1)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        distill_features = []
        # print(type(x))
        # print(type(x[0]))
        distill_features.append(self.conv0(x[0]))
        distill_features.append(self.conv1(x[1]))
        distill_features.append(self.conv2(x[2]))
        distill_features.append(self.conv3(x[3]))
        distill_features.append(self.conv4(x[4]))
        if self.is_s:
            distill_features.append(self.m * x[5])
        if self.is_m:
            distill_features.append(self.m * x[5][self.m_no])

        return distill_features

    # ---------------- xavier fan_in----------------
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                value = math.sqrt(3. / n)
                m.weight.data.uniform_(-value, value)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.in_features
                value = math.sqrt(3. / n)
                m.weight.data.uniform_(-value, value)
                if m.bias is not None:
                    m.bias.data.zero_()

def distill_models(input_channel=[64, 64, 128, 256, 512],
                   output_channel=[64, 64, 128, 256, 512], **kwargs):
    distill_modules = EncoderModule(input_channel, output_channel, **kwargs)
    return distill_modules

def source_resnet18(channel_nums, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_s(BasicBlock, BasicBlock_last, channel_nums, [2, 2, 2, 2], **kwargs)

    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        # params = torch.load('models/resnet50-19c8e357.pth')
        pretrained_model = './weights/resnet/resnet18-5c106cde.pth'
        print('load pretrained model from: {}'.format(pretrained_model))
        params = torch.load(pretrained_model)
        del params['fc.weight']
        del params['fc.bias']
        model.load_state_dict(params, strict=False)
    return model

def target_resnet18(**kwargs):

    model = ResNet_t(BasicBlock, BasicBlock_last, [2, 2, 2, 2], **kwargs)

    return model

def component_resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # model = ResNet(BasicBlock, BasicBlock_last, [2, 2, 2, 2], **kwargs)
    model = ResNet_c(BasicBlock, BasicBlock_last, [2, 2, 2, 2], **kwargs)

    return model
