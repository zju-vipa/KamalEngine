from tqdm import tqdm
import torch.nn.functional as F 
import torch
from kamal.core import metrics

class Evaluator(object):
    def __init__(self, metric, dataloader):
        self.dataloader = dataloader
        self.metric = metric

    def eval(self, model, device=None, progress=False):
        self.metric.reset()
        with torch.no_grad():
            for i, (inputs, targets) in enumerate( tqdm(self.dataloader, disable=not progress) ):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model( inputs )
                self.metric.update(outputs, targets)
        return self.metric.get_results()
    
    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)

class AdvEvaluator(object):
    def __init__(self, metric, dataloader, adversary):
        self.dataloader = dataloader
        self.metric = metric
        self.adversary = adversary

    def eval(self, model, device=None, progress=False):
        self.metric.reset()
        for i, (inputs, targets) in enumerate( tqdm(self.dataloader, disable=not progress) ):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = self.adversary.perturb(inputs, targets)
            with torch.no_grad():
                outputs = model( inputs )
                self.metric.update(outputs, targets)
        return self.metric.get_results()
    
    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)

def classification_evaluator(dataloader):
    metric = metrics.MetricCompose({
        'Acc': metrics.TopkAccuracy(),
        'Loss': metrics.RunningLoss(torch.nn.CrossEntropyLoss(reduction='sum'))
    })
    return Evaluator( metric, dataloader=dataloader)

def advarsarial_classification_evaluator(dataloader, adversary):
    metric = metrics.MetricCompose({
        'Acc': metrics.TopkAccuracy(),
        'Loss': metrics.RunningLoss(torch.nn.CrossEntropyLoss(reduction='sum'))
    })
    return AdvEvaluator( metric, dataloader=dataloader, adversary=adversary)


def segmentation_evaluator(dataloader, num_classes, ignore_idx=255):
    cm = metrics.ConfusionMatrix(num_classes, ignore_idx=ignore_idx)
    metric = metrics.MetricCompose({
        'mIoU': metrics.mIoU(cm),
        'Acc': metrics.Accuracy(),
        'Loss': metrics.RunningLoss(torch.nn.CrossEntropyLoss(reduction='sum'))
    })
    return Evaluator( metric, dataloader=dataloader)