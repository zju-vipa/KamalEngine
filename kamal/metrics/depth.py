from .stream_metrics import _StreamMetrics
import numpy as np

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

    def get_results(self, return_key_metric=False):
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

        if return_key_metric:
            return ('absolute relative', ard)

        return {
            'absolute relative': ard,
            'squared relative': srd,
            'percents within thresholds': threshold_percents
        }

    def reset(self):
        self.preds = []
        self.targets = []
