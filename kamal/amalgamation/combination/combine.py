import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

def _assert_same_type(layers, layer_type=None):
    if layer_type is None:
        layer_type = type(layers[0])

    assert all(isinstance(l, layer_type) for l in layers), 'Model archictures must be the same'

def _get_layers(model_list):
    submodel = [ model.modules() for model in model_list ]
    for layers in zip(*submodel):
        _assert_same_type(layers)
        yield layers

def bn_combine_fn(layers):
    """Combine 2D Batch Normalization Layers
    
    **Parameters:**
        - **layers** (BatchNorm2D): Batch Normalization Layers.
    """
    _assert_same_type(layers, nn.BatchNorm2d)
    num_features = sum(l.num_features for l in layers)
    combined_bn = nn.BatchNorm2d(num_features=num_features,
                                 eps=layers[0].eps,
                                 momentum=layers[0].momentum,
                                 affine=layers[0].affine,
                                 track_running_stats=layers[0].track_running_stats)
    combined_bn.running_mean = torch.cat(
        [l.running_mean for l in layers], dim=0).clone()
    combined_bn.running_var = torch.cat(
        [l.running_var for l in layers], dim=0).clone()

    if combined_bn.affine:
        combined_bn.weight = torch.nn.Parameter(
            torch.cat([l.weight.data.clone() for l in layers], dim=0).clone())
        combined_bn.bias = torch.nn.Parameter(
            torch.cat([l.bias.data.clone() for l in layers], dim=0).clone())
    return combined_bn


def conv2d_combine_fn(layers):
    """Combine 2D Conv Layers
    
    **Parameters:**
        - **layers** (Conv2d): Conv Layers.
    """
    _assert_same_type(layers, nn.Conv2d)

    CO, CI = 0, 0
    for l in layers:
        O, I, H, W = l.weight.shape
        CO += O
        CI += I

    dtype = layers[0].weight.dtype
    device = layers[0].weight.device

    combined_weight = torch.nn.Parameter(
        torch.zeros(CO, CI, H, W, dtype=dtype, device=device))
    #combined_weight.data = torch.torch.zeros( CO,CI,H,W, dtype=dtype, device=device)

    if layers[0].bias is not None:
        combined_bias = torch.nn.Parameter(
            torch.zeros(CO, dtype=dtype, device=device))
    else:
        combined_bias = None

    co_offset = 0
    ci_offset = 0

    for idx, l in enumerate(layers):
        co_len, ci_len = l.weight.shape[0], l.weight.shape[1]

        combined_weight[co_offset: co_offset+co_len,
                        ci_offset: ci_offset+ci_len, :, :] = l.weight.clone()

        if combined_bias is not None:
            combined_bias[co_offset: co_offset+co_len] = l.bias.clone()
        co_offset += co_len
        ci_offset += ci_offset

    combined_conv2d = nn.Conv2d(in_channels=CI,
                                out_channels=CO,
                                kernel_size=layers[0].weight.shape[-2:],
                                stride=layers[0].stride,
                                padding=layers[0].padding,
                                bias=layers[0].bias)

    combined_conv2d.weight.data = combined_weight
    if combined_bias is not None:
        combined_conv2d.bias.data = combined_bias

    for p in combined_conv2d.parameters():
        p.requires_grad = True

    return combined_conv2d


def combine_models(models):
    """Combine modules with parser
    
    **Parameters:**
        - **models** (nn.Module): modules to be combined.
        - **combine_parser** (function): layer selector
    """

    def _recursively_combine(module):
        module_output = module

        if isinstance( module, nn.Conv2d ):
            combined_module = conv2d_combine_fn( layer_mapping[module] )
        elif isinstance( module, nn.BatchNorm2d ):
            combined_module = bn_combine_fn( layer_mapping[module] )
        
        if combined_module is not None:
            module_output = combined_module

        for name, child in module.named_children():
            module_output.add_module(name, _recursively_combine(child))
        return module_output

    teacher_models = deepcopy(teacher_models)
    combined_model = deepcopy(teacher_models[0]) # copy the model archicture and modify it with _recursively_combine
    layer_mapping = {}
    for combined_layer, layers in zip(combined_model.modules(), _get_layers(teacher_models)):
        layer_mapping[combined_layer] = layers  # link to teachers
    combined_model = _recursively_combine(combined_model)
    return combined_model


class CombinedModel(nn.Module):
    def __init__(self, models):
        super( Combination, self ).__init__()
        self.combined_model = combine_models( models )
        self.expand = len(models)

    def forward(self, x):
        x.repeat( -1, x.shape[1]*self.expand, -1, -1 )
        return self.combined_model(x)
        

