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

import abc
import torch
import torch.nn as nn
import torch.nn.functional as F 
import sys
import typing
from typing import Callable, Dict, List, Any
from collections import Mapping, Sequence
from . import criterions
from kamal.engine import metrics, exceptions, evaluator

class predefined_metrics(object):

    @staticmethod
    def classification():
        return metrics.MetricCompose({
            'Acc': metrics.TopkAccuracy(),
            'Loss': metrics.RunningAverage(torch.nn.CrossEntropyLoss(reduction='mean'))
        })

    @staticmethod
    def segmentation(num_classes, ignore_idx=255):
        cm = metrics.ConfusionMatrix(num_classes, ignore_idx=ignore_idx)
        return metrics.MetricCompose({
            'mIoU': metrics.mIoU(cm),
            'Acc': metrics.Accuracy(),
            'Loss': metrics.RunningAverage(torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=ignore_idx))
        })

    @staticmethod
    def regression():
        return metrics.MetricCompose(
            metric_dict={'mse': metrics.MeanSquaredError()}
        )


    @staticmethod
    def monocular_depth():
        return metrics.MetricCompose(
            metric_dict={
                'rmse': metrics.RootMeanSquaredError(),
                'rmse_log': metrics.RootMeanSquaredError( log_scale=True, ),
                'rmse_scale_inv': metrics.ScaleInveriantMeanSquaredError(),
                'abs rel': metrics.AbsoluteRelativeDifference(),
                'sq rel': metrics.SquaredRelativeDifference(), 
                'percents within thresholds': metrics.Threshold( thresholds=[1.25, 1.25**2, 1.25**3],  )
            }
        )
        
    @staticmethod
    def loss_metric(loss_fn):
        return metrics.MetricCompose(
            metric_dict={
                'loss': metrics.AverageMetric( loss_fn )
            }
        )

class predefined_evaluator():

    @staticmethod
    def classification_evaluator(dataloader):
        metric = predefined_metrics.classification()
        return evaluator.Evaluator( metric, dataloader=dataloader)

    @staticmethod
    def segmentation_evaluator(dataloader, num_classes, ignore_idx=255):
        metric = predefined_metrics.segmentation(num_classes=num_classes, ignore_idx=ignore_idx)
        return evaluator.Evaluator( metric, dataloader=dataloader)