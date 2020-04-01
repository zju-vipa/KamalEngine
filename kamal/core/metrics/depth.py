from .stream_metrics import StreamMetricsBase
import numpy as np

class StreamDepthMetrics(object):
    PRIMARY_METRIC = 'rmse'

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
        masks = (self.targets != self.ignore_index)
        count = np.sum(masks)
        self.targets = self.targets[masks].clip(1e-3)
        self.preds = self.preds[masks].clip(1e-3)


        diff = np.abs(self.targets - self.preds)
        diff_log = np.log( self.targets ) - np.log( self.preds )

        rmse = np.sqrt( (diff**2).mean() )
        rmse_log = np.sqrt( (diff_log**2).mean() )

        rmse_scale_inv = ( (diff_log**2).sum() / count - \
            0.5 * (diff_log.sum() / count)**2 )
        
        sigma = np.maximum(self.targets / self.preds, self.preds / self.targets)
    
        ard = diff / self.targets
        ard = np.sum(ard) / count

        srd = diff * diff / self.targets
        srd = np.sum(srd) / count

        threshold_percents = {}
        for threshold in self.thresholds:
            threshold_percents[threshold] = np.nansum((sigma < threshold)) / count
        
        return {
            'rmse': float(rmse),
            'rmse_log': float(rmse_log),
            'rmse_scale_inv': float(rmse_scale_inv),
            'ard': float(ard),
            'srd': float(srd),
            'percents within thresholds': threshold_percents
        }

    def reset(self):
        self.preds = []
        self.targets = []
