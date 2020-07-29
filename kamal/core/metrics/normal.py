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
from kamal.core.metrics.stream_metrics import StreamMetricsBase

class NormalPredictionMetrics(StreamMetricsBase):

    @property
    def PRIMARY_METRIC(self):
        return 'mean angle'

    def __init__(self, thresholds):
        self.thresholds = thresholds
        self.preds = None
        self.targets = None
        self.masks = None

    @torch.no_grad()
    def update(self, preds, targets, masks):
        """
        **Type**: numpy.ndarray or torch.Tensor
        **Shape:**
            - **preds**: $(N, 3, H, W)$.
            - **targets**: $(N, 3, H, W)$.
            - **masks**: $(N, 1, H, W)$.
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
            targets = targets.cpu().numpy()
            masks = masks.cpu().numpy()

        self.preds = preds if self.preds is None else np.append(self.preds, preds, axis=0)
        self.targets = targets if self.targets is None else np.append(self.targets, targets, axis=0)
        self.masks = masks if self.masks is None else np.append(self.masks, masks, axis=0)

    def get_results(self):
        """
        **Returns:**
            - **mean angle**
            - **median angle**
            - **precents for angle within thresholds**
        """
        products = np.sum(self.preds * self.targets, axis=1)

        angles = np.arccos(np.clip(products, -1.0, 1.0)) / np.pi * 180
        self.masks = self.masks.squeeze(1)
        angles = angles[self.masks == 1]

        mean_angle = np.mean(angles)
        median_angle = np.median(angles)
        count = self.masks.sum()

        threshold_percents = {}
        for threshold in self.thresholds:
            # threshold_percents[threshold] = np.sum((angles < threshold)) / count
            threshold_percents[threshold] = np.mean(angles < threshold)

        if return_key_metric:
            return ('absolute relative', ard)

        return {
            'mean angle': mean_angle,
            'median angle': median_angle,
            'percents within thresholds': threshold_percents
        }

    def reset(self):
        self.preds = None
        self.targets = None
        self.masks = None