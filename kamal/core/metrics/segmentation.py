from sklearn.metrics import confusion_matrix
import numpy as np
import torch

from kamal.core.metrics.stream_metrics import StreamMetricsBase

class SegmentationMetrics(StreamMetricsBase):
    """
    Stream Metrics for Semantic Segmentation Task
    """
    @property
    def PRIMARY_METRIC(self):
        return 'mIoU'

    def __init__(self, n_classes, ignore_index=255):
        self.ignore_index=255
        self.n_classes = n_classes
        self.reset()

    @torch.no_grad()
    def update(self, outputs, targets):
        preds = outputs.max(1)[1]
        preds, targets = preds.detach().cpu().numpy(), targets.detach().cpu().numpy()
        
        for p, t in zip(preds, targets):
            self.confusion_matrix += self._fast_hist( p.flatten(), t.flatten() )
    
    def _fast_hist(self, pred, target):
        mask = (target!=self.ignore_index)
        hist = np.bincount(
            self.n_classes * target[mask].astype(int) + pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self, class_iou=False):
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        miou = np.nanmean(iu)
        cls_iu = dict(zip(range(self.n_classes), iu))

        results =  {
                "acc": acc,
                "mIoU": miou,
            }
        if class_iou:
            results["class IoU"] = cls_iu
        return results

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))