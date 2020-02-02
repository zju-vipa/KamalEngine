import abc
import torch
import torch.nn as nn
import torch.nn.functional as F 

class TaskBase(abc.ABC):
    def __init__(self, criterion):
        self.criterion = criterion

    @abc.abstractmethod
    def get_loss(self, model, inputs, targets):
        pass

    @abc.abstractmethod
    def inference(self, model, inputs):
        pass 

class ClassificationTask(TaskBase):
    def __init__(self, criterion=nn.CrossEntropyLoss(ignore_index=255)):
        super(ClassificationTask, self).__init__(criterion)

    def get_loss(self, model, inputs, targets): 
        logits = model( inputs )
        loss = self.criterion(logits, targets.squeeze())
        return {'loss': loss}

    def inference(self, model, inputs):
        logits = model( inputs )
        preds = logits.max(1)[1]
        return {'preds': preds}

class SegmentationTask(ClassificationTask):
    def __init__( self, criterion=nn.CrossEntropyLoss() ):
        super(SegmentationTask, self).__init__(criterion)

class ReconstructionTask( TaskBase ):
    def __init__(self, criterion=nn.MSELoss()):
        super(ReconstructionTask, self).__init__(criterion)

    def get_loss(self, model, inputs, targets): 
        outputs = model( inputs )
        loss = self.criterion(logits, outputs)
        return {'loss': loss}

    def inference(self, model, inputs):
        outputs = model( inputs )
        return {'preds': preds}
    