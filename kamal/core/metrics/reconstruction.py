from .stream_metrics import StreamMetricsBase
from pytorch_msssim import ssim, ms_ssim
from ..loss.functional import psnr

class StreamReconstructionMetrics(StreamMetricsBase):
    PRIMARY_METRIC = 'ms-ssim'
    def __init__(self, data_range=1.0):
        self._psnr = 0.0
        self._ms_ssim = 0.0
        self.cnt = 0
        self.data_range = data_range

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

    def to_str(self, result):
        string = "Psnr: %.4f\nMS-SSIM: %.4f" % (result['psnr'], result['ms-ssim'])
        return string

    def reset(self):
        self._psnr = 0.0
        self._ms_ssim = 0.0
        self.cnt = 0