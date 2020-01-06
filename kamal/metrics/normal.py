from .stream_metrics import _StreamMetrics
import numpy as np

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

    def get_results(self, return_key_metric=False):
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