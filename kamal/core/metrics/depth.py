import numpy as np
import torch

from kamal.core.metrics.stream_metrics import StreamMetricsBase

class DepthEstimationMetrics(StreamMetricsBase):

    @property
    def PRIMARY_METRIC(self):
        return 'rmse'

    def __init__(self, thresholds):
        self.thresholds = thresholds
        self.reset()

    @torch.no_grad()
    def update(self, preds, targets):
        """
        **Type**: numpy.ndarray or torch.Tensor
        **Shape:**
            - **preds**: $(N, H, W)$. 
            - **targets**: $(N, H, W)$. 
        """
        preds, targets = preds.clamp(min=1e-6), targets.clamp(min=1e-6)
        
        diff = torch.abs(preds - targets)
        diff_log = torch.log( preds ) - torch.log( targets )

        self._cnt += torch.numel(diff)

        # RMSE
        self._accum_sq_diff += (diff**2).sum()

        # scale-invariant log-RMSE
        self._accum_log_diff = diff_log.sum()
        self._accum_sq_log_diff = (diff_log**2).sum()

        # relative difference
        self._accum_abs_rel += (diff/targets).sum()
        self._accum_sq_rel += ((diff**2)/targets).sum()

        # Threshold
        sigma = torch.max(preds / targets, targets / preds)
        for thres in self.thresholds:
            self._accum_thres[thres]+=torch.sum( sigma<thres )

    def get_results(self):
        """
        **Returns:**
            - **absolute relative error**
            - **squared relative error**
            - **precents for $r$ within thresholds**: Where $r_i = max(preds_i/targets_i, targets_i/preds_i)$
        """
        return {
            'rmse': torch.sqrt( self._accum_sq_diff / self._cnt ).item(),
            'rmse_log': torch.sqrt( self._accum_sq_log_diff / self._cnt ).item(),
            'rmse_scale_inv': ( self._accum_sq_log_diff / self._cnt - 0.5 * (self._accum_log_diff**2 / self._cnt**2) ).item(),
            'abs rel': (self._accum_abs_rel / self._cnt).item(),
            'sq rel': (self._accum_sq_rel / self._cnt).item(),
            'percents within thresholds': {
                    thres: (self._accum_thres[thres] / self._cnt).item() for thres in self.thresholds
                }
        }
    
    def reset(self):
        self._accum_sq_diff = 0.
        self._accum_log_diff = 0.
        self._accum_sq_log_diff = 0.
        self._accum_abs_rel = 0.
        self._accum_sq_rel = 0.
        self._accum_thres = {thres: 0. for thres in self.thresholds}
        self._cnt = 0.