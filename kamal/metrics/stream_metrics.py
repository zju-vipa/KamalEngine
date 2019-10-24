from __future__ import division
import torch

import numpy as np
from ..misc.ssim import ms_ssim
from ..misc.psnr import psnr

from ..misc.ssim_np import MultiScaleSSIM as ms_ssim_np
from skimage.measure import compare_psnr

from sklearn.metrics import confusion_matrix


class MetrcisCompose(object):
    def __init__(self, metrics):
        self.metrics = metrics

    def __call__(self, preds, targets):
        self.update(preds, targets)

    def update(self, preds, targets):
        for metric, pred, target in zip(self.metrics, preds, targets):
            if metric is None:
                continue

            metric.update(pred.detach(), target.detach())

    def reset(self):
        for metric in self.metrics:
            metric.reset()

    def get_results(self):
        return [metric.get_results() for metric in self.metrics]

    def to_str(self, results):
        return [metric.to_str(result) for metric, result in zip(self.metrics, results)]


class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def __call__(self, pred, target):
        self.update(pred, target)

    def update(self, pred, target):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()


class StreamCompMetrics(_StreamMetrics):
    def __init__(self, data_range=1.0):
        self._psnr = 0.0
        self._ms_ssim = 0.0
        self.cnt = 0
        self.data_range = data_range

    def update(self, pred, target):
        assert len(pred.shape) == 4, pred.shape

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1).clip(0, self.data_range)
        target = target.cpu().numpy().transpose(0, 2, 3, 1)

        for img_test, img_true in zip(pred, target):
            self._psnr += compare_psnr(img_true,
                                       img_test, data_range=self.data_range)
            self._ms_ssim += ms_ssim_np(np.expand_dims(img_true, axis=0),
                                        np.expand_dims(img_test, axis=0), max_val=self.data_range)
        self.cnt += pred.shape[0]

    def get_results(self):
        return {
            "PSNR": self._psnr / self.cnt,
            "MS-SSIM": self._ms_ssim / self.cnt
        }

    def to_str(self, result):
        string = "Psnr: %f\nMS-SSIM: %f" % (result['PSNR'], result['MS-SSIM'])
        return string

    def reset(self):
        self._psnr = 0.0
        self._ms_ssim = 0.0
        self.cnt = 0


class StreamClsMetrics(_StreamMetrics):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, pred, target):
        pred = pred.max(dim=1)[1].cpu().numpy().astype(np.uint8)
        target = target.cpu().numpy().astype(np.uint8)
        for lt, lp in zip(target, pred):
            self.confusion_matrix[lt][lp] += 1

    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k != "Class IoU":
                string += "%s: %f\n" % (k, v)
        #string+='Class IoU:\n'
        # for k, v in results['Class IoU'].items():
        #    string += "\tclass %d: %f\n"%(k, v)
        return string

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) +
                              hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {
            "Overall Acc": acc,
            "Mean Acc": acc_cls,
            "FreqW Acc": fwavacc,
            "Mean IoU": mean_iu,
            "Class IoU": cls_iu
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class StreamSegMetrics(_StreamMetrics):
    def __init__(self, n_classes, ignore_index=255):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.ignore_index = ignore_index

    def update(self, preds, targets):
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
            targets = targets.cpu().numpy()

        if len(preds.shape) == 4:
            preds = preds.argmax(axis=1)

        for lt, lp in zip(targets, preds):
            self.confusion_matrix += self._fast_hist(
                lt.flatten(), lp.flatten())

    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k != "Class IoU":
                string += "%s: %f\n" % (k, v)

        string += 'Class IoU:\n'
        for k, v in results['Class IoU'].items():
            string += "\tclass %d: %f\n" % (k, v)
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes) & (
            label_true != self.ignore_index)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) +
                              hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {
            "Overall Acc": acc,
            "Mean Acc": acc_cls,
            "FreqW Acc": fwavacc,
            "Mean IoU": mean_iu,
            "Class IoU": cls_iu,
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class AverageMeter(object):
    """ Average Record
    """

    def __init__(self):
        self.record = dict()

    def update(self, **kargs):
        for k, v in kargs.items():
            rec = self.record.get(k, None)

            if rec is None:
                self.record[k] = {'val': v, 'count': 1}  # val, count
            else:
                rec['val'] += v
                rec['count'] += 1

    def get_results(self, *args):
        return {k: (self.record[k]['val']/self.record[k]['count']) for k in args}
        # return [ (self.record[k]['val']/self.record[k]['count']) for k in args ]

    def reset(self, *args):
        for k in args:
            self.record[k] = {'val': 0.0, 'count': 0}


class StreamDepthMetrics(_StreamMetrics):
    """This metric is used in depth prediction task.

    **Parameters:**
        - **thresholds** (list of float)
        - **ignore_index** (int, optional): Value to ignore.
    """
    def __init__(self, thresholds, ignore_index=0):
        self.thresholds = thresholds
        self.ignore_index = ignore_index
        self.preds = []
        self.targets = []

    def update(self, preds, targets):
        """
        **Type**: numpy.ndarray or torch.Tensor
        **Shape:**
            - **preds**: $(N, H, W)$. 
            - **targets**: $(N, H, W)$. 
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
            targets = targets.cpu().numpy()

        self.preds = np.append(self.preds, preds)
        self.targets = np.append(self.targets, targets)

    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k!="percents within thresholds":
                string += "%s: %f\n"%(k, v)
        
        string+='percents within thresholds:\n'
        for k, v in results['percents within thresholds'].items():
            string += "\tthreshold %f: %f\n"%(k, v)
        return string

    def get_results(self):
        """
        **Returns:**
            - **absolute relative error**
            - **squared relative error**
            - **precents for $r$ within thresholds**: Where $r_i = max(preds_i/targets_i, targets_i/preds_i)$
        """
        masks = self.targets != self.ignore_index
        count = np.sum(masks)
        self.targets = self.targets[masks]
        self.preds = self.preds[masks]

        diff = np.abs(self.targets - self.preds)
        sigma = np.maximum(self.targets / self.preds, self.preds / self.targets)

        ard = diff / self.targets
        ard = np.sum(ard) / count

        srd = diff * diff / self.targets
        srd = np.sum(srd) / count

        threshold_percents = {}
        for threshold in self.thresholds:
            threshold_percents[threshold] = np.nansum((sigma < threshold)) / count

        return {
            'absolute relative': ard,
            'squared relative': srd,
            'percents within thresholds': threshold_percents
        }

    def reset(self):
        self.preds = []
        self.targets = []

class StreamAngleMetrics(_StreamMetrics):
    """This metric is used in surface normal prediction task.

    **Parameters:**
        - **thresholds** (list of float)
    """
    def __init__(self, thresholds):
        self.thresholds = thresholds
        self.preds = None
        self.targets = None
        self.masks = None

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

    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k!="percents within thresholds":
                string += "%s: %f\n"%(k, v)
        
        string+='percents within thresholds:\n'
        for k, v in results['percents within thresholds'].items():
            string += "\tthreshold %f: %f\n"%(k, v)
        return string

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

        return {
            'mean angle': mean_angle,
            'median angle': median_angle,
            'percents within thresholds': threshold_percents
        }

    def reset(self):
        self.preds = None
        self.targets = None
        self.masks = None

class CEMAPMetric():
    def __init__(self):
        self.targets = None
        self.preds = None

    def update(self, preds, targets):
        # targets: -1 negative, 0 difficult, 1 positive
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
            targets = targets.cpu().numpy()

        self.preds = preds if self.preds is None else np.append(self.preds, preds, axis=0)
        self.targets = targets if self.targets is None else np.append(self.targets, targets, axis=0)

    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            string += '{}: {}\n'.format(k, v)

        return string

    def get_results(self):
        nTest = self.targets.shape[0]
        nLabel = self.targets.shape[1]
        eap = np.zeros(nTest)
        for i in range(0,nTest):
            R = np.sum(self.targets[i,:]==1)
            for j in range(0,nLabel):            
                if self.targets[i,j]==1:
                    r = np.sum(self.preds[i,np.nonzero(self.targets[i,:]!=0)]>=self.preds[i,j])
                    rb = np.sum(self.preds[i,np.nonzero(self.targets[i,:]==1)] >= self.preds[i,j])

                    eap[i] = eap[i] + rb/(r*1.0)
            eap[i] = eap[i]/R
        # emap = np.nanmean(ap)


        cap = np.zeros(nLabel)
        for i in range(0,nLabel):
            R = np.sum(self.targets[:,i]==1)
            for j in range(0,nTest):
                if self.targets[j,i]==1:
                    r = np.sum(self.preds[np.nonzero(self.targets[:,i]!=0),i] >= self.preds[j,i])
                    rb = np.sum(self.preds[np.nonzero(self.targets[:,i]==1),i] >= self.preds[j,i])
                    cap[i] = cap[i] + rb/(r*1.0)
            cap[i] = cap[i]/R
        # cmap = np.nanmean(ap)

        return {
            'eap': eap,
            'emap': np.nanmean(eap),
            'cap': cap,
            'cmap': np.nanmean(cap),
        }

    def reset(self):
        self.preds = None
        self.targets = None
        