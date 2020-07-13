from .stream_metrics import Metric
import torch
from typing import Callable

class ConfusionMatrix(Metric):
    def __init__(self, num_classes, ignore_idx=None, attach_to=None):
        super(ConfusionMatrix, self).__init__(attach_to=attach_to)
        self._num_classes = num_classes   
        self._ignore_idx = ignore_idx     
        self.reset()

    @torch.no_grad()
    def update(self, outputs, targets):
        outputs, targets = self._attach(outputs, targets)
        if self.confusion_matrix.device != outputs.device:
            self.confusion_matrix = self.confusion_matrix.to(device=outputs.device)
        preds = outputs.max(1)[1].flatten()
        targets = targets.flatten()
        mask = (preds<self._num_classes) & (preds>=0)
        if self._ignore_idx:
            mask = mask & (targets!=self._ignore_idx)
        preds, targets = preds[mask], targets[mask]
        hist = torch.bincount( self._num_classes * targets + preds, 
                minlength=self._num_classes ** 2 ).view(self._num_classes, self._num_classes)
        self.confusion_matrix += hist

    def get_results(self):
        return self.confusion_matrix.detach().cpu()

    def reset(self):
        self._cnt = 0
        self.confusion_matrix = torch.zeros(self._num_classes, self._num_classes, dtype=torch.int64, requires_grad=False)

class IoU(Metric):
    def __init__(self, confusion_matrix: ConfusionMatrix, attach_to=None):
        self._confusion_matrix = confusion_matrix

    def update(self, outputs, targets):
        pass # update will be done in confusion matrix

    def reset(self):
        pass
        
    def get_results(self):
        cm = self._confusion_matrix.get_results()
        iou = cm.diag() / (cm.sum(dim=1) + cm.sum(dim=0) - cm.diag() + 1e-9)
        return iou

class mIoU(IoU):
    def get_results(self):
        return super(mIoU, self).get_results().mean()
