from pytorch_msssim import ssim, ms_ssim

from kamal.core.metrics.stream_metrics import StreamMetricsBase
from kamal.core.criterion.functional import psnr
import torch

class ReconstructionMetrics(StreamMetricsBase):
    @property
    def PRIMARY_METRIC(self):
        return 'ms-ssim'
        
    def __init__(self, data_range=1.0):
        self._psnr = 0.0
        self._ms_ssim = 0.0
        self.cnt = 0
        self.data_range = data_range

    @torch.no_grad()
    def update(self, preds, targets):
        assert len(preds.shape) == 4, preds.shape
        psnr_results += psnr(preds, targets, data_range=self.data_range, size_average=False).detach().cpu().numpy().sum()
        msssim_results += ms_ssim( preds, targets, data_range=self.data_range, size_average=False).detach().cpu().numpy().sum()
        self.cnt += preds.shape[0]

    def get_results(self):
        return {
            "psnr": self._psnr / self.cnt,
            "ms-ssim": self._ms_ssim / self.cnt
        }

    def reset(self):
        self._psnr = 0.0
        self._ms_ssim = 0.0
        self.cnt = 0