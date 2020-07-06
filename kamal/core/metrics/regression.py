import numpy as np
import torch

from kamal.core.metrics.stream_metrics import Metric
from typing import Callable

__all__=['MeanSquaredError', 'RootMeanSquaredError', 'MeanAbsoluteError', 
         'ScaleInveriantMeanSquaredError', 'RelativeDifference', 
         'AbsoluteRelativeDifference', 'SquaredRelativeDifference', 'Threshold' ]

class MeanSquaredError(Metric):
    def __init__(self, output_target_transform: Callable=lambda x,y: (x.view_as(y), y) ):
        super(MeanSquaredError, self).__init__( output_target_transform=output_target_transform )
        self.reset()

    @torch.no_grad()
    def update(self, ouputs, targets):
        ouputs, targets = self._output_target_transform(ouputs, targets)
        diff = torch.sum((ouputs - targets)**2)
        self._accum_sq_diff += diff
        self._cnt += torch.numel(ouputs)

    def get_results(self):
        return (self._accum_sq_diff / self._cnt).detach().cpu()
    
    def reset(self):
        self._accum_sq_diff = 0.
        self._cnt = 0.


class RootMeanSquaredError(MeanSquaredError):
    def get_results(self):
        return torch.sqrt( (self._accum_sq_diff / self._cnt) ).detach().cpu()


class MeanAbsoluteError(Metric):
    def __init__(self, output_target_transform: Callable=lambda x,y: (x.view_as(y), y) ):
        super(MeanAbsoluteError, self).__init__( output_target_transform=output_target_transform )
        self.reset()

    @torch.no_grad()
    def update(self, ouputs, targets):
        ouputs, targets = self._output_target_transform(ouputs, targets)
        diff = torch.sum((ouputs - targets).abs())
        self._accum_abs_diff += diff
        self._cnt += torch.numel(ouputs)

    def get_results(self):
        return (self._accum_abs_diff / self._cnt).detach().cpu()
    
    def reset(self):
        self._accum_abs_diff = 0.
        self._cnt = 0.


class ScaleInveriantMeanSquaredError(Metric):
    def __init__(self, output_target_transform: Callable=lambda x,y: (x.view_as(y), y) ):
        super(ScaleInveriantMeanSquaredError, self).__init__( output_target_transform=output_target_transform )
        self.reset()

    @torch.no_grad()
    def update(self, ouputs, targets):
        ouputs, targets = self._output_target_transform(ouputs, targets)
        diff_log = torch.log( ouputs+1e-8 ) - torch.log( targets+1e-8 )
        self._accum_log_diff = diff_log.sum()
        self._accum_sq_log_diff = (diff_log**2).sum()
        self._cnt += torch.numel(ouputs)

    def get_results(self):
        return ( self._accum_sq_log_diff / self._cnt - 0.5 * (self._accum_log_diff**2 / self._cnt**2) ).detach().cpu()
    
    def reset(self):
        self._accum_log_diff = 0.
        self._accum_sq_log_diff = 0.
        self._cnt = 0.


class RelativeDifference(Metric):
    def __init__(self, output_target_transform: Callable=lambda x,y: (x.view_as(y), y) ):
        super(RelativeDifference, self).__init__( output_target_transform=output_target_transform )
        self.reset()

    @torch.no_grad()
    def update(self, ouputs, targets):
        ouputs, targets = self._output_target_transform(ouputs, targets)
        diff = (ouputs - targets).abs()
        self._accum_abs_rel += (diff/targets).sum()
        self._cnt += torch.numel(ouputs)

    def get_results(self):
        return (self._accum_abs_rel / self._cnt).detach().cpu()
    
    def reset(self):
        self._accum_abs_rel = 0.
        self._cnt = 0.


class AbsoluteRelativeDifference(Metric):
    def __init__(self, output_target_transform: Callable=lambda x,y: (x.view_as(y), y) ):
        super(AbsoluteRelativeDifference, self).__init__( output_target_transform=output_target_transform )
        self.reset()

    @torch.no_grad()
    def update(self, ouputs, targets):
        ouputs, targets = self._output_target_transform(ouputs, targets)
        diff = (ouputs - targets).abs()
        self._accum_abs_rel += (diff/targets).sum()
        self._cnt += torch.numel(ouputs)

    def get_results(self):
        return (self._accum_abs_rel / self._cnt).detach().cpu()
    
    def reset(self):
        self._accum_abs_rel = 0.
        self._cnt = 0.


class AbsoluteRelativeDifference(Metric):
    def __init__(self, output_target_transform: Callable=lambda x,y: (x.view_as(y), y) ):
        super(AbsoluteRelativeDifference, self).__init__( output_target_transform=output_target_transform )
        self.reset()

    @torch.no_grad()
    def update(self, ouputs, targets):
        ouputs, targets = self._output_target_transform(ouputs, targets)
        diff = (ouputs - targets).abs()
        self._accum_abs_rel += (diff/targets).sum()
        self._cnt += torch.numel(ouputs)

    def get_results(self):
        return (self._accum_abs_rel / self._cnt).detach().cpu()
    
    def reset(self):
        self._accum_abs_rel = 0.
        self._cnt = 0.


class SquaredRelativeDifference(Metric):
    def __init__(self, output_target_transform: Callable=lambda x,y: (x.view_as(y), y) ):
        super(SquaredRelativeDifference, self).__init__( output_target_transform=output_target_transform )
        self.reset()

    @torch.no_grad()
    def update(self, ouputs, targets):
        ouputs, targets = self._output_target_transform(ouputs, targets)
        diff = (ouputs - targets)**2
        self._accum_sq_rel += (diff/targets).sum()
        self._cnt += torch.numel(ouputs)

    def get_results(self):
        return (self._accum_sq_rel / self._cnt).detach().cpu()
    
    def reset(self):
        self._accum_sq_rel = 0.
        self._cnt = 0.


class Threshold(Metric):
    def __init__(self, thresholds=[1.25, 1.25**2, 1.25**3], output_target_transform: Callable=lambda x,y: (x.view_as(y), y) ):
        super(Threshold, self).__init__( output_target_transform=output_target_transform )
        self.thresholds = thresholds
        self.reset()
        

    @torch.no_grad()
    def update(self, ouputs, targets):
        ouputs, targets = self._output_target_transform(ouputs, targets)
        sigma = torch.max(ouputs / targets, targets / ouputs)
        for thres in self.thresholds:
            self._accum_thres[thres]+=torch.sum( sigma<thres )
        self._cnt += torch.numel(ouputs)

    def get_results(self):
        return { thres: (self._accum_thres[thres] / self._cnt).detach().cpu() for thres in self.thresholds }
    
    def reset(self):
        self._cnt = 0.
        self._accum_thres = {thres: 0. for thres in self.thresholds}