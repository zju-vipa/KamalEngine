from .stream_metrics import _StreamMetrics
import numpy as np

class StreamClassificationMetrics(_StreamMetrics):
    def __init__(self):
        self.correct = 0
        self.total = 0

    def update(self, pred, target):
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        self.correct += (pred==target).sum()
        self.total += len(target)

    @staticmethod
    def to_str(results):
        return "%s: %.4f"%( 'Acc', results['acc'] )

    def get_results(self, return_key_metric=False):
        if return_key_metric:
            return ('acc', self.correct / self.total)
        return {"acc": self.correct / self.total }
    
    def reset(self):
        self.correct = self.total = 0

class StreamCEMAPMetrics():
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

    def get_results(self, return_key_metric=False):
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
        if return_key_metric:
            return ('eap', eap)

        return {
            'eap': eap,
            'emap': np.nanmean(eap),
            'cap': cap,
            'cmap': np.nanmean(cap),
        }

    def reset(self):
        self.preds = None
        self.targets = None
        