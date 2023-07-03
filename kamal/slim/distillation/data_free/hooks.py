import torch
import torch.nn as nn
import torch.nn.functional as F

def register_hooks(modules):
    hooks = []
    for m in modules:
        hooks.append( FeatureHook(m) )
    return hooks

class InstanceMeanHook(object):
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module

    def hook_fn(self, module, input, output):
        self.instance_mean = torch.mean(input[0], dim=[2, 3])

    def remove(self):
        self.hook.remove()

    def __repr__(self):
        return "<Feature Hook>: %s"%(self.module)

class FeatureHook(object):
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module

    def hook_fn(self, module, input, output):
        self.output = output
        self.input = input[0]
    
    def remove(self):
        self.hook.remove()

    def __repr__(self):
        return "<Feature Hook>: %s"%(self.module)


class FeatureMeanHook(object):
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module

    def hook_fn(self, module, input, output):
        self.instance_mean = torch.mean(input[0], dim=[2, 3])

    def remove(self):
        self.hook.remove()

    def __repr__(self):
        return "<Feature Hook>: %s"%(self.module)


class FeatureMeanVarHook():
    def __init__(self, module, on_input=True, dim=[0,2,3]):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.on_input = on_input
        self.module = module
        self.dim = dim

    def hook_fn(self, module, input, output):
        # To avoid inplace modification
        if self.on_input:
            feature = input[0].clone() 
        else:
            feature = output.clone()
        self.var, self.mean = torch.var_mean( feature, dim=self.dim, unbiased=True )

    def remove(self):
        self.hook.remove()
        self.output=None


class DeepInversionHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module, mmt_rate):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module
        self.mmt_rate = mmt_rate
        self.mmt = None
        self.tmp_val = None

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        # forcing mean and variance to match between two distributions
        # other ways might work better, i.g. KL divergence
        if self.mmt is None:
            r_feature = torch.norm(module.running_var.data - var, 2) + \
                        torch.norm(module.running_mean.data - mean, 2)
        else:
            mean_mmt, var_mmt = self.mmt
            r_feature = torch.norm(module.running_var.data - (1 - self.mmt_rate) * var - self.mmt_rate * var_mmt, 2) + \
                        torch.norm(module.running_mean.data - (1 - self.mmt_rate) * mean - self.mmt_rate * mean_mmt, 2)

        self.r_feature = r_feature
        self.tmp_val = (mean, var)

    def update_mmt(self):
        mean, var = self.tmp_val
        if self.mmt is None:
            self.mmt = (mean.data, var.data)
        else:
            mean_mmt, var_mmt = self.mmt
            self.mmt = ( self.mmt_rate*mean_mmt+(1-self.mmt_rate)*mean.data,
                         self.mmt_rate*var_mmt+(1-self.mmt_rate)*var.data )

    def remove(self):
        self.hook.remove()
