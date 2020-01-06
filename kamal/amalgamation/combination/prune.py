import torch
import torch.nn as nn
import torch.nn.functional as F


def prune_bn_layer(layer, idxs):
    """Prune batch normalization layers
    
    **Parameters:**
        - **layer** (BatchNorm2d): BN layer.
        - **idxs** (list or tuple): index of pruned weights.
    """
    assert isinstance(layer, nn.BatchNorm2d)
    if isinstance(idxs, int):
        idxs = [idxs]
    num_pruned = len(idxs)

    keep_idxs = [i for i in range(layer.num_features) if i not in idxs]
    layer.num_features = layer.num_features-num_pruned

    layer.running_mean = layer.running_mean.data.clone()[keep_idxs]
    layer.running_var = layer.running_var.data.clone()[keep_idxs]

    if layer.affine:
        layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])
        layer.bias = torch.nn.Parameter(layer.bias.data.clone()[keep_idxs])


def prune_related_conv_layer(layer, idxs):
    """Prune the influenced conv2d layer
    
    **Parameters:**
        - **layer** (BatchNorm2d): conv layer.
        - **idxs** (list or tuple): index of pruned weights.
    """

    assert isinstance(layer, nn.Conv2d)
    if isinstance(idxs, int):
        idxs = [idxs]
    num_pruned = len(idxs)
    keep_idxs = [i for i in range(layer.in_channels) if i not in idxs]
    layer.in_channels = layer.in_channels - num_pruned
    layer.weight = torch.nn.Parameter(
        layer.weight.data.clone()[:, keep_idxs, :, :])


def prune_conv_layer(layer, idxs):
    """Prune conv2d layer
    
    **Parameters:**
        - **layer** (BatchNorm2d): conv layer.
        - **idxs** (list or tuple): index of pruned weights.
    """

    assert isinstance(layer, nn.Conv2d)
    if isinstance(idxs, int):
        idxs = [idxs]
    num_pruned = len(idxs)
    keep_idxs = [i for i in range(layer.out_channels) if i not in idxs]

    layer.out_channels = layer.out_channels-num_pruned

    if layer.bias is not None:
        old_bias = layer.bias.data.clone()

    layer.weight = torch.nn.Parameter(
        layer.weight.data.clone()[keep_idxs, :, :, :])
    if layer.bias is not None:
        layer.bias = torch.nn.Parameter(layer.bias.data.clone()[keep_idxs])
