from .darknet import *
from .mobilenetv2 import mobile_half
from .resnet import *
from .resnetv2 import *
from .vgg import *
from .gan import *
from . import cifar
from .vgg_block import *
from .resnet_customize import *
from typing import Dict
from .alexnet import alexnet
from .shufflenet import *
from .shufflenetv2 import *
from .LEARNTOBRANCH import *
from . import lenet, wresnet, vgg, resnet, mobilenetv2, shufflenetv2, resnet_tiny, resnet_in
from .resnet_ import resnet8, resnet14, resnet20, resnet32, resnet44, resnet56, resnet110, resnet8x4, resnet32x4
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2

MODEL_DICT = {
    'resnet8': resnet8,
    'resnet14': resnet14,
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'resnet110': resnet110,
    'resnet8x4': resnet8x4,
    'resnet32x4': resnet32x4,
    'ResNet18': ResNet18,
    'ResNet34': ResNet34,
    'ResNet50': ResNet50,
    'ResNet101': ResNet101,
    'ResNet152': ResNet152,
    'wrn_16_1': wrn_16_1,
    'wrn_16_2': wrn_16_2,
    'wrn_40_1': wrn_40_1,
    'wrn_40_2': wrn_40_2,
    'vgg11': vgg11_bn,
    'vgg13': vgg13_bn,
    'vgg16': vgg16_bn,
    'vgg19': vgg19_bn,
    'MobileNetV2': mobile_half,
    'ShuffleV1': ShuffleV1,
    'ShuffleV2': ShuffleNetV2,
}


def get_model(model_name: str, num_classes: int, state_dict: Dict[str, torch.Tensor] = None, **kwargs):
    fn = MODEL_DICT[model_name]
    model = fn(num_classes=num_classes, **kwargs)

    if state_dict is not None:
        model.load_state_dict(state_dict)
    return model
