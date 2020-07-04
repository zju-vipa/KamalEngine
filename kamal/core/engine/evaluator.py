import abc, sys
import torch
from tqdm import tqdm

from kamal.core.engine import task
from kamal.core import metrics
from kamal.utils import set_mode

class EvaluatorBase(abc.ABC):
    
    @property
    def PRIMARY_METRIC(self):
        return self.metric.PRIMARY_METRIC

    def __init__(self, data_loader, metric, task):
        self.data_loader = data_loader
        self.task = task
        self.metric = metric

    @abc.abstractmethod
    def eval(self, model):
        pass

class ClassificationEvaluator(EvaluatorBase):
    def __init__(self, 
                data_loader, 
                metric=None,
                task=task.ClassificationTask(),
                progress=False):
        if metric is None:
            metric = metrics.ClassificationMetrics()
        super(ClassificationEvaluator, self).__init__(data_loader, metric, task)
        self.progress = progress

    def eval(self, model, device=None, postprocess=None):
        device = device if device is not None else \
            torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        self.metric.reset()
        model.to(device)
        with torch.no_grad(), set_mode(model, training=False):
            for i, (inputs, targets) in enumerate( tqdm(self.data_loader, disable=not self.progress) ): 
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.task.get_outputs( model, inputs )
                if postprocess is not None:
                    outputs = postprocess( outputs )
                self.metric.update( outputs, targets )
        return self.metric.get_results()


class SegmentationEvaluator(ClassificationEvaluator):
    def __init__(self, 
                 num_classes, 
                 data_loader, 
                 metric=None,
                 task=task.SegmentationTask(), 
                 progress=False):
        if metric is None:
            metric = metrics.SegmentationMetrics(num_classes, ignore_index=255)
        super( SegmentationEvaluator, self ).__init__(data_loader=data_loader, metric=metric, task=task, progress=progress)


class DepthEvaluator(EvaluatorBase):
    def __init__(self, 
                data_loader, 
                metric=None,
                task=task.MonocularDepthTask(),
                progress=False):
        if metric is None:
            metric = metrics.DepthEstimationMetrics(thresholds=[1.25, 1.25**2, 1.25**3])
        super(DepthEvaluator, self).__init__(data_loader, metric, task)
        self.progress = progress
    
    def eval(self, model, device=None):
        device = device if device is not None else \
            torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        self.metric.reset()
        model.to(device)
        with torch.no_grad(), set_mode(model, training=False):
            for i, (inputs, targets) in enumerate( tqdm(self.data_loader, disable=not self.progress) ):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.task.get_outputs( model, inputs )
                self.metric.update(outputs, targets)
        return self.metric.get_results()


class CriterionEvaluator(EvaluatorBase):

    @property
    def PRIMARY_METRIC(self):
        return 'loss'
    
    def __init__(self, data_loader, task, progress=False):
        super(CriterionEvaluator, self).__init__(data_loader, None, task)
        self.progress = progress

    def eval(self, model, device=None):
        device = device if device is not None else \
            torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        model.to(device)
        avg_loss = 0
        with torch.no_grad(), set_mode(model, training=False):
            for i, (inputs, targets) in enumerate( tqdm(self.data_loader, disable=not self.progress) ): 
                inputs, targets = inputs.to(device), targets.to(device)
                loss = sum(self.task.get_loss( model, inputs, targets ))
                avg_loss+=loss.item()
        return {'loss': (avg_loss / len(self.data_loader))}


class MultitaskEvaluator(EvaluatorBase):
    def __init__(self, data_loader, task, metric_list, progress=False):
        self.task = task
        self.data_loader = data_loader
        self.metric_list = metric_list
        self.progress = progress

    def eval(self, model, device=None):
        device = device if device is not None else \
            torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        model.to(device)

        for m in self.metric_list:
            m.reset()

        with torch.no_grad(), set_mode(model, training=False):
            for i, (images, targets_list) in enumerate( tqdm(self.data_loader, disable=not self.progress) ):
                images = images.to(device)
                outputs_list = self.task.get_outputs( model, images )

                for j, (out, target, metric) in enumerate(zip(outputs_list, targets_list, self.metric_list)):
                    targets = targets.to(device)
                    metric.update(out, target)
        
        results = dict()
        for metric in zip(self.metric_list):
            results.update( metrics.get_results() )
        return results

class SbmEvaluator(MultitaskEvaluator):
    def __init__(self, data_loader, split_size, tasks, task=task.SbmTask(), progress=False):
        super(SbmEvaluator, self).__init__(data_loader, split_size, tasks, task=task, progress=progress)

class KDClassificationEvaluator(EvaluatorBase):
    def __init__(self, 
                data_loader, 
                teacher,
                task=task.ClassificationTask(),
                progress=False):
        super(KDClassificationEvaluator, self).__init__(data_loader, task)
        self.metrics = metrics.ClassificationMetrics()
        self.progress = progress
        self.teacher = teacher

    def eval(self, model, device=None):
        device = device if device is not None else \
            torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        self.metrics.reset()
        teacher = self.teacher

        model.to(device)
        teacher.to(device)
        
        with torch.no_grad(), \
             set_mode(model, training=False), \
             set_mode( teacher, training=False ): 
            for i, data in enumerate( tqdm(self.data_loader, disable=not self.progress) ): 
                inputs = data[0] if isinstance( data, (tuple, list) ) else data
                inputs = inputs.to(device)
                s_preds = self.task.predict( model, inputs )['preds']
                t_preds = self.task.predict( teacher, inputs )['preds']
                self.metrics.update( s_preds, t_preds )
        return self.metrics.get_results()

