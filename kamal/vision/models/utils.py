# A synchronized version modified from https://github.com/pytorch/vision
import os, sys
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.hub import *

def _get_torch_home():
    torch_home = os.path.expanduser(
        os.getenv(ENV_TORCH_HOME,
                  os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'torch')))
    return torch_home

def download_from_url(url, model_dir=None, map_location=None, progress=True, check_hash=False):
    r"""
    Adapted from torchvision.models.utils.load_state_dict_from_url
    This function only download files from the specified url and return its path as a string.
    It is used to get weight files in other formats.
    """

    # Issue warning to move data if old env is set
    if os.getenv('TORCH_MODEL_ZOO'):
        warnings.warn('TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead')

    if model_dir is None:
        torch_home = _get_torch_home()
        model_dir = os.path.join(torch_home, 'checkpoints')

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = HASH_REGEX.search(filename).group(1) if check_hash else None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)
    
    return cached_file


def load_darknet_weights( model, darknet_file):
    layers_with_params = [ layer for layer in model.modules() \
        if isinstance( layer, (nn.Conv2d, nn.Linear, nn.BatchNorm2d) ) ]
    
    with open(darknet_file, 'rb') as fp:
        major, minor, revision = np.fromfile(fp, dtype=np.int32, count=3)
        if major*10 + minor >= 2 and major < 1000 and minor < 1000:
            seen = np.fromfile(fp, dtype=np.int64, count=1)
        else:
            seen = np.fromfile(fp, dtype=np.int32, count=1)
        #transpose = (major > 1000) | (minor > 1000);

        weights = np.fromfile( fp, dtype=np.float32 )

        offset = 0
        for i, layer in enumerate( layers_with_params ) :
            if isinstance( layer, nn.Conv2d ):
                conv = layer
                if i<len(layers_with_params)-1 and isinstance(layers_with_params[i+1], nn.BatchNorm2d):
                    bn = layers_with_params[i+1]
                    num_bias = bn.bias.numel()
                    # load
                    bn_bias = torch.from_numpy(weights[offset:offset+num_bias])
                    offset+=num_bias
                    bn_weight = torch.from_numpy(weights[offset:offset+num_bias])
                    offset+=num_bias
                    bn_running_mean = torch.from_numpy(weights[offset:offset+num_bias])
                    offset+=num_bias
                    bn_running_var = torch.from_numpy(weights[offset:offset+num_bias])
                    offset+=num_bias

                    # reshape
                    bn_bias = bn_bias.view_as(bn.bias)
                    bn_weight = bn_weight.view_as(bn.weight)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # copy
                    bn.bias.data.copy_( bn_bias )
                    bn.weight.data.copy_( bn_weight )
                    bn.running_mean.data.copy_( bn_running_mean )
                    bn.running_var.data.copy_( bn_running_var )
                else: # conv without bn
                    num_bias = conv.bias.numel()
                    conv_bias = torch.from_numpy( weights[offset: offset+num_bias] ).view_as(conv.bias)
                    conv.bias.data.copy_( conv_bias )
                    offset+=num_bias

                num_weight = conv.weight.numel()
                # n, c, h, w = conv.weight.shape
                conv_weight = torch.from_numpy(weights[offset:offset+num_weight])  #.view_as(conv.weight)
                # conv_weight = conv_weight.view_as(n, h, w, c).permute(0, 3, 1, 2).contiguous()
                conv_weight = conv_weight.view_as(conv.weight)
                conv.weight.data.copy_( conv_weight )
                offset+=num_weight
            
            elif isinstance( layer, nn.Linear ):
                linear = layer
                num_bias, num_weight = linear.weight.numel(), linear.bias.numel()
                linear_bias = torch.from_numpy( weights[offset: offset+num_bias] ).view_as( linear.bias )
                offset+=num_bias
                linear_weight = torch.from_numpy( weights[offset: offset+num_weight] ).view_as( linear.weight )
                offset+=num_weight
                linear.bias.data.copy_( linear_bias )
                linear.weight.data.copy_( linear_weight )

class Hint(nn.Module):
	'''
	FitNets: Hints for Thin Deep Nets
	https://arxiv.org/pdf/1412.6550.pdf
	'''
	def __init__(self):
		super(Hint, self).__init__()

	def forward(self, fm_s, fm_t):
		loss = F.mse_loss(fm_s, fm_t)

		return loss