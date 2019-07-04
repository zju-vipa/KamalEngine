# coding:utf-8
import os
import torch
import math
import pdb
from torch import nn
from torch.autograd import Variable
from torch import optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

from .alexnet import alexnet

use_cuda = True

outputs = []


def obtain_features(module, input, output):
    outputs.append(output.data)


def add_alexnet_hook(model):
    for conv in model.features:
        conv.register_forward_hook(obtain_features)

    for fc in model.classifier:
        fc.register_forward_hook(obtain_features)


class InnerLayer:

    def __init__(self, reallayer):
        self.reallayer = reallayer
        self.initialized = False
        self.databuffer = None
        self.reallayer.register_forward_hook(self.obtain_middle_result)

    def __call__(self, data):
        if self.initialized == False:
            self._initialize_weight()
            self.initialized = True
        if use_cuda:
            self.reallayer = self.reallayer.cuda()
        return self.reallayer(data)

    def _initialize_weight(self):
        if isinstance(self.reallayer, nn.Conv2d):
            n = self.reallayer.kernel_size[0] * \
                self.reallayer.kernel_size[1] * self.reallayer.in_channels
            value = math.sqrt(3. / n)
            self.reallayer.weight.data.uniform_(-value, value)
            if self.reallayer.bias is not None:
                self.reallayer.bias.data.zero_()
        elif isinstance(self.reallayer, nn.BatchNorm2d):
            self.reallayer.weight.data.fill_(1)
            self.reallayer.bias.data.zero_()
        elif isinstance(self.reallayer, nn.Linear):
            n = self.reallayer.in_features
            value = math.sqrt(3. / n)
            self.reallayer.weight.data.uniform_(-value, value)
            if self.reallayer.bias is not None:
                self.reallayer.bias.data.zero_()

    def get_run_result(self):
        return self.databuffer

    def obtain_middle_result(self, module, input, output):
        self.databuffer = output.data

    def parameters(self):
        return self.reallayer.parameters()

    def save(self):
        return self.reallayer.state_dict()

    def load(self, st):
        self.reallayer.load_state_dict(st)


class InnerBlock(nn.Module):
    def __init__(self, layerlist):
        super(InnerBlock, self).__init__()
        self.block = nn.ModuleList(layerlist)

    def forward(self, x):
        for l in self.block:
            if use_cuda:
                l = l.cuda()
            if type(l) == torch.nn.modules.linear.Linear:
                x = pyreshape(x)
            x = l(x)
        return x


def parse_layer(torchlayer):
    tmp = InnerLayer(torchlayer)
    tmp.initialized = True
    return tmp


def pyreshape(data):
    data = data.view(data.size(0), -1)
    return data


def get_innerlayer(typename, args=None):
    if typename == "Conv2d":
        if "in_channels" not in args or "out_channels" not in args or "kernel_size" not in args:
            print("layer initialize parameter error")
            return
        return InnerLayer(nn.Conv2d(args['in_channels'], args['out_channels'], args['kernel_size'], stride=args['stride'], padding=args['padding']))

    if typename == "Linear":
        if "in_channels" not in args or "out_channels" not in args:
            print("layer initialize parameter error")
            return
        return InnerLayer(nn.Linear(args['in_channels'], args['out_channels']))

    if typename == "ReLU":
        return InnerLayer(nn.ReLU(inplace=False))

    if typename == "MaxPool2d":
        return InnerLayer(nn.MaxPool2d(kernel_size=3, stride=2))

    if typename == "Dropout":
        return InnerLayer(nn.Dropout())


def get_loss(name, **args):
    if name == "MSE":
        return nn.MSELoss(size_average=True)
    else:
        print("loss name not supported")


def get_optimizer(name, param, lr, momentum, weight_decay):
    if name == "SGD":
        return optim.SGD(param, lr, momentum, weight_decay)
    else:
        print("optimizer name not supported")


def fetch_params(t_param_path):

    teacher = alexnet(pretrained=False, num_classes=60)
    teacher.load_state_dict(torch.load(t_param_path))
    add_alexnet_hook(teacher)
    if torch_is_cuda():
        teacher = teacher.cuda()
    return teacher


def concat_feature(featurelist):
    resultfeature = None
    for feature in featurelist:
        if type(resultfeature) == torch.cuda.FloatTensor or type(resultfeature) == torch.FloatTensor:
            resultfeature = torch.cat((resultfeature, feature), dim=1)
        else:
            resultfeature = feature
    return resultfeature


def get_variable(tensor, volatile=False):
    return Variable(tensor, volatile)


def torch_load(path):
    return torch.load(path)


def torch_concat(a, b, dim=1):
    return torch.cat((a, b), dim=1)


def torch_save(to_save, path):
    torch.save(to_save, path)


def torch_is_cuda():
    return torch.cuda.is_available()


def get_cuda(input):
    return input.cuda()
