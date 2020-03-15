import abc
import torch
import torch.nn as nn
import torch.nn.functional as F 
from ..loss import KDLoss

class TaskBase(abc.ABC):
    def __init__(self, criterion):
        self.criterion = criterion

    @abc.abstractmethod
    def get_loss(self, model, inputs, targets):
        pass

    @abc.abstractmethod
    def predict(self, model, inputs):
        pass 

class ClassificationTask(TaskBase):
    def __init__(self, criterion=nn.CrossEntropyLoss(ignore_index=255)):
        super(ClassificationTask, self).__init__(criterion)

    def get_loss(self, model, inputs, targets): 
        logits = model( inputs )
        loss = self.criterion(logits, targets.squeeze())
        return {'loss': loss}

    def predict(self, model, inputs):
        logits = model( inputs )
        # multiple output
        if isinstance(logits, (tuple, list)):
            logits = logits[-1]
        preds = logits.max(1)[1]
        return {'preds': preds}

class SegmentationTask(ClassificationTask):
    def __init__( self, criterion=nn.CrossEntropyLoss(ignore_index=255) ):
        super(SegmentationTask, self).__init__(criterion)

class ReconstructionTask( TaskBase ):
    def __init__(self, criterion=nn.MSELoss()):
        super(ReconstructionTask, self).__init__(criterion)

    def get_loss(self, model, inputs, targets): 
        outputs = model( inputs )
        loss = self.criterion(logits, outputs)
        return {'loss': loss}

    def predict(self, model, inputs):
        outputs = model( inputs )
        return {'preds': preds}

class DepthTask( ReconstructionTask ):
    def __init__(self, criterion=nn.L1Loss()):
        super(DepthTask, self).__init__(criterion)

class KDClassificationTask(ClassificationTask):
    def __init__(self, criterion=KDLoss(use_kldiv=True)):
        super(KDClassificationTask, self).__init__(criterion)
    
    def get_loss( self, model, teacher, inputs ):
        s_logits = model( inputs )
        t_logits = teacher( inputs )
        loss = self.criterion( s_logits, t_logits )
        return {'loss': loss }

class KDSegmentationTask(SegmentationTask):
    def __init__( self, criterion=KDLoss() ):
        super(KDSegmentationTask, self).__init__(criterion)