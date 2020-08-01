# Copyright 2020 Zhejiang Lab. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================

import numpy as np
import torch

from kamal.core.metrics.stream_metrics import Metric
from typing import Callable

__all__=['MeanSquaredError', 'RootMeanSquaredError', 'MeanAbsoluteError', 
         'ScaleInveriantMeanSquaredError', 'RelativeDifference', 
         'AbsoluteRelativeDifference', 'SquaredRelativeDifference', 'Threshold' ]

class MeanSquaredError(Metric):
    def __init__(self, log_scale=False, attach_to=None):
        super(MeanSquaredError, self).__init__( attach_to=attach_to )
        self.reset()
        self.log_scale=log_scale

    @torch.no_grad()
    def update(self, outputs, targets):
        outputs, targets = self._attach(outputs, targets)
        if self.log_scale:
            diff = torch.sum((torch.log(outputs+1e-8) - torch.log(targets+1e-8))**2)
        else:
            diff = torch.sum((outputs - targets)**2)
        self._accum_sq_diff += diff
        self._cnt += torch.numel(outputs)

    def get_results(self):
        return (self._accum_sq_diff / self._cnt).detach().cpu()
    
    def reset(self):
        self._accum_sq_diff = 0.
        self._cnt = 0.


class RootMeanSquaredError(MeanSquaredError):
    def get_results(self):
        return torch.sqrt( (self._accum_sq_diff / self._cnt) ).detach().cpu()


class MeanAbsoluteError(Metric):
    def __init__(self, attach_to=None ):
        super(MeanAbsoluteError, self).__init__( attach_to=attach_to )
        self.reset()

    @torch.no_grad()
    def update(self, outputs, targets):
        outputs, targets = self._attach(outputs, targets)
        diff = torch.sum((outputs - targets).abs())
        self._accum_abs_diff += diff
        self._cnt += torch.numel(outputs)

    def get_results(self):
        return (self._accum_abs_diff / self._cnt).detach().cpu()
    
    def reset(self):
        self._accum_abs_diff = 0.
        self._cnt = 0.


class ScaleInveriantMeanSquaredError(Metric):
    def __init__(self, attach_to=None ):
        super(ScaleInveriantMeanSquaredError, self).__init__( attach_to=attach_to )
        self.reset()

    @torch.no_grad()
    def update(self, outputs, targets):
        outputs, targets = self._attach(outputs, targets)
        diff_log = torch.log( outputs+1e-8 ) - torch.log( targets+1e-8 )
        self._accum_log_diff = diff_log.sum()
        self._accum_sq_log_diff = (diff_log**2).sum()
        self._cnt += torch.numel(outputs)

    def get_results(self):
        return ( self._accum_sq_log_diff / self._cnt - 0.5 * (self._accum_log_diff**2 / self._cnt**2) ).detach().cpu()
    
    def reset(self):
        self._accum_log_diff = 0.
        self._accum_sq_log_diff = 0.
        self._cnt = 0.


class RelativeDifference(Metric):
    def __init__(self, attach_to=None ):
        super(RelativeDifference, self).__init__( attach_to=attach_to )
        self.reset()

    @torch.no_grad()
    def update(self, outputs, targets):
        outputs, targets = self._attach(outputs, targets)
        diff = (outputs - targets).abs()
        self._accum_abs_rel += (diff/targets).sum()
        self._cnt += torch.numel(outputs)

    def get_results(self):
        return (self._accum_abs_rel / self._cnt).detach().cpu()
    
    def reset(self):
        self._accum_abs_rel = 0.
        self._cnt = 0.


class AbsoluteRelativeDifference(Metric):
    def __init__(self, attach_to=None ):
        super(AbsoluteRelativeDifference, self).__init__( attach_to=attach_to )
        self.reset()

    @torch.no_grad()
    def update(self, outputs, targets):
        outputs, targets = self._attach(outputs, targets)
        diff = (outputs - targets).abs()
        self._accum_abs_rel += (diff/targets).sum()
        self._cnt += torch.numel(outputs)

    def get_results(self):
        return (self._accum_abs_rel / self._cnt).detach().cpu()
    
    def reset(self):
        self._accum_abs_rel = 0.
        self._cnt = 0.


class AbsoluteRelativeDifference(Metric):
    def __init__(self, attach_to=None ):
        super(AbsoluteRelativeDifference, self).__init__( attach_to=attach_to )
        self.reset()

    @torch.no_grad()
    def update(self, outputs, targets):
        outputs, targets = self._attach(outputs, targets)
        diff = (outputs - targets).abs()
        self._accum_abs_rel += (diff/targets).sum()
        self._cnt += torch.numel(outputs)

    def get_results(self):
        return (self._accum_abs_rel / self._cnt).detach().cpu()
    
    def reset(self):
        self._accum_abs_rel = 0.
        self._cnt = 0.


class SquaredRelativeDifference(Metric):
    def __init__(self, attach_to=None ):
        super(SquaredRelativeDifference, self).__init__( attach_to=attach_to )
        self.reset()

    @torch.no_grad()
    def update(self, outputs, targets):
        outputs, targets = self._attach(outputs, targets)
        diff = (outputs - targets)**2
        self._accum_sq_rel += (diff/targets).sum()
        self._cnt += torch.numel(outputs)

    def get_results(self):
        return (self._accum_sq_rel / self._cnt).detach().cpu()
    
    def reset(self):
        self._accum_sq_rel = 0.
        self._cnt = 0.


class Threshold(Metric):
    def __init__(self, thresholds=[1.25, 1.25**2, 1.25**3], attach_to=None ):
        super(Threshold, self).__init__( attach_to=attach_to )
        self.thresholds = thresholds
        self.reset()
        

    @torch.no_grad()
    def update(self, outputs, targets):
        outputs, targets = self._attach(outputs, targets)
        sigma = torch.max(outputs / targets, targets / outputs)
        for thres in self.thresholds:
            self._accum_thres[thres]+=torch.sum( sigma<thres )
        self._cnt += torch.numel(outputs)

    def get_results(self):
        return { thres: (self._accum_thres[thres] / self._cnt).detach().cpu() for thres in self.thresholds }
    
    def reset(self):
        self._cnt = 0.
        self._accum_thres = {thres: 0. for thres in self.thresholds}