from .stream_metrics import _StreamMetrics
from pytorch_msssim import ssim, ms_ssim
from ..loss.functional import psnr

class StreamReconstructionMetrics(_StreamMetrics):
    def __init__(self, data_range=1.0):
        self._psnr = 0.0
        self._ms_ssim = 0.0
        self.cnt = 0
        self.data_range = data_range

    def update(self, pred, target):
        assert len(pred.shape) == 4, pred.shape
        psnr_results += psnr(pred, target, data_range=self.data_range, size_average=False).detach().cpu().numpy().sum()
        msssim_results += ms_ssim( pred, target, data_range=self.data_range, size_average=False).detach().cpu().numpy().sum()
        self.cnt += pred.shape[0]

    def get_results(self, return_key_metric=False):
        if return_key_metric:
            return ('ms-ssim', self._ms_ssim / self.cnt)
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