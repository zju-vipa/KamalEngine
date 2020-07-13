import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from typing import Callable

from kamal.core.engine.engine import Engine
from kamal.core.engine.trainer import KDTrainer
from kamal.core.engine.hooks import FeatureHook
from kamal.core import tasks
import math

from kamal.slim.prunning import Pruner, strategy

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
                                bias=layers[0].bias is not None)
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
        else:
            combined_module = module

        if combined_module is not None:
            module_output = combined_module

        for name, child in module.named_children():
            module_output.add_module(name, _recursively_combine(child))
        return module_output

    models = deepcopy(models)
    combined_model = deepcopy(models[0]) # copy the model archicture and modify it with _recursively_combine

    layer_mapping = {}
    for combined_layer, layers in zip(combined_model.modules(), _get_layers(models)):
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


class PruningKDTrainer(KDTrainer):
    def setup(
        self,
        student,
        teachers,
        task,
        dataloader:  torch.utils.data.DataLoader, 
        get_optimizer_and_scheduler:Callable=None, 
        pruning_rounds=5,
        device=None,
    ):
        if device is None:
            device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        self._device = device
        self._dataloader = dataloader
        self.model = self.student = student.to(self.device)
        self.teachers = nn.ModuleList(teachers).to(self.device) 
        self.get_optimizer_and_scheduler = get_optimizer_and_scheduler
    
    def run(self, max_iter, start_iter=0, epoch_length=None, pruning_rounds=3, target_model_size=0.6 ):
        pruning_size_per_round = 1 - math.pow( target_model_size, 1/pruning_rounds )
        prunner = Pruner( strategy.LNStrategy(n=1) )
        for pruning_round in range(pruning_rounds):
            prunner.prune( self.student, rate=pruning_size_per_round, example_inputs=torch.randn(1,3,240,240) )
            self.student.to(self.device)
            if self.get_optimizer_and_scheduler:
                self.optimizer, self.scheduler = self.get_optimizer_and_scheduler( self.student )
            else:
                self.optimizer = torch.optim.Adam( self.student.parameters(), lr=1e-4, weight_decay=1e-5 )
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( self.optimizer, T_max= (max_iter-start_iter)//pruning_rounds )
            step_iter = (max_iter - start_iter)//pruning_rounds

            with set_mode(self.student, training=True), \
                set_mode(self.teachers, training=False):
                super( RecombinationAmalgamation, self ).run(self.step_fn, self._dataloader, 
                        start_iter=start_iter+step_iter*pruning_round , max_iter=start_iter+step_iter*(pruning_round+1), epoch_length=epoch_length)
    
    def step_fn(self, engine, batch):
        metrics = super(RecombinationAmalgamation, self).step_fn( engine, batch )
        self.scheduler.step()
        return metrics

class RecombinationAmalgamator(PruningKDTrainer):
    pass