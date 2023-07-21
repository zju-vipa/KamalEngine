from typing import Dict

import torch

from .resnet import resnet8, resnet14, resnet20, resnet32, resnet44, resnet56, resnet110, resnet8x4, resnet32x4
from .resnetv2 import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
import torch.nn as nn


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
}


def get_model(model_name: str, num_classes: int, state_dict: Dict[str, torch.Tensor] = None, **kwargs):
    fn = MODEL_DICT[model_name]
    model = fn(num_classes=num_classes, **kwargs)

    if state_dict is not None:
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model
