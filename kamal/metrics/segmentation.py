from .stream_metrics import _StreamMetrics
from sklearn.metrics import confusion_matrix
import numpy as np
import torch

class StreamSegmentationMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, n_classes, ignore_index=255):
        self.ignore_index=255
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, preds, targets):
        if isinstance(preds, torch.Tensor):
            preds, targets = preds.detach().cpu().numpy(), targets.detach().cpu().numpy()
            
        for p, t in zip(preds, targets):
            self.confusion_matrix += self._fast_hist( p.flatten(), t.flatten() )
    
    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k!="Class IoU":
                string += "%s: %f\n"%(k, v)
        return string

    def _fast_hist(self, pred, target):
        mask = (target!=self.ignore_index)
        hist = np.bincount(
            self.n_classes * target[mask].astype(int) + pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self, return_key_metric=False):
        hist = self.confusion_matrix

        acc = np.diag(hist).sum() / hist.sum()
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        miou = np.nanmean(iu)
        cls_iu = dict(zip(range(self.n_classes), iu))
        if return_key_metric:
            return ('mIoU', miou)

        return {
                "acc": acc,
                "mIoU": miou,
                "class IoU": cls_iu, 
            }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))