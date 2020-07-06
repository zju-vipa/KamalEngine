import numpy as np
import torch
from kamal.core.metrics.stream_metrics import Metric
from typing import Callable

__all__=['Accuracy', 'TopkAccuracy']

class Accuracy(Metric):
    def __init__(self, output_target_transform: Callable=lambda x,y: (x,y)):
        super(Accuracy, self).__init__(output_target_transform=output_target_transform)
        self.reset()

    @torch.no_grad()
    def update(self, outputs, targets):
        outputs, targets = self._output_target_transform(outputs, targets)
        outputs = outputs.max(1)[1]
        self._correct += ( outputs.view(-1)==targets.view(-1) ).sum()
        self._cnt += torch.numel( targets )

    def get_results(self):
        return (self._correct / self._cnt).detach().cpu()
    
    def reset(self):
        self._correct = self._cnt = 0.0


class TopkAccuracy(Metric):
    def __init__(self, topk=5, output_target_transform: Callable=lambda x,y: (x,y)):
        super(TopkAccuracy, self).__init__(output_target_transform=output_target_transform)
        self._topk = topk
        self.reset()
    
    @torch.no_grad()
    def update(self, outputs, targets):
        outputs, targets = self._output_target_transform(outputs, targets)
        _, outputs = outputs.topk(self._topk, dim=1, largest=True, sorted=True)
        correct = outputs.eq( targets.view(-1, 1).expand_as(outputs) )
        self._correct += correct[:, :self._topk].view(-1).float().sum(0).item()
        self._cnt += len(targets)
    
    def get_results(self):
        return self._correct / self._cnt

    def reset(self):
        self._correct = 0.0
        self._cnt = 0.0


class StreamCEMAPMetrics():
    @property
    def PRIMARY_METRIC(self):
        return "eap"

    def __init__(self):
        self.reset()

    def update(self, logits, targets):
        preds = logits.max(1)[1]
        # targets: -1 negative, 0 difficult, 1 positive
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
            targets = targets.cpu().numpy()

        self.preds = preds if self.preds is None else np.append(self.preds, preds, axis=0)
        self.targets = targets if self.targets is None else np.append(self.targets, targets, axis=0)

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