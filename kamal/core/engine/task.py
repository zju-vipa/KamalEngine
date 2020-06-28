import abc
import torch
import torch.nn as nn
import torch.nn.functional as F 
import sys
import typing

class TaskBase(abc.ABC):
    def __init__(self, criterions, weights=None):
        if not isinstance( criterions, typing.Sequence ):
            criterions = [ criterions ]
        self.criterions = criterions
        self.weights = weights
    
    def preprocess(self, inputs): 
        return inputs

    def postprocess(self, outputs):
        return outputs

    def get_outputs(self, model, inputs):
        inputs = self.preprocess( inputs )
        return model(inputs)

    def get_predictions(self, model, inputs):
        outputs = self.get_outputs( model, inputs )
        return self.postprocess( outputs )
    
    def get_loss(self, model, inputs, targets):
        outputs = self.get_outputs( model, inputs )
        if isinstance( targets, (list, tuple) ): 
            if isinstance( outputs, (list, tuple) ): # Multi-Task
                assert len(outputs)==len(self.criterions) and len(outputs)==len(targets)
                loss = [ criterion( output, target ) for (criterion, output, target) in zip( self.criterions, outputs, targets ) ]
            elif isinstance( outputs, torch.Tensor ): # Multi-Loss   
                loss = [ criterion( outputs, target ) for (criterion, target) in zip( self.criterions, targets ) ]
        else: # simple training
            loss = [ self.criterions[0](outputs, targets) ]
        if self.weights is not None:
            assert len(self.weights)==len(loss)
            loss = [ l*w for (l, w) in zip( loss, self.weights ) ]
        return loss


class ClassificationTask(TaskBase):
    def __init__(self, criterions=nn.CrossEntropyLoss(), weights=None ):
        super(ClassificationTask, self).__init__(criterions, weights)

    def postprocess(self, outputs):
        return outputs.max(1)[1]


class SegmentationTask(ClassificationTask):
    def __init__( self, criterions=nn.CrossEntropyLoss(ignore_index=255), weights=None ):
        super(SegmentationTask, self).__init__(criterions, weights)


class MonocularDepthTask( TaskBase ):
    def __init__(self, criterions=nn.MSELoss(), weights=None):
        super(MonocularDepthTask, self).__init__(criterions, weights)

    def get_loss(self, model, inputs, targets):
        outputs = self.get_outputs( model, inputs )
        if isinstance( targets, (list, tuple) ): 
            if isinstance( outputs, (list, tuple) ): # Multi-Task
                assert len(outputs)==len(self.criterions) and len(outputs)==len(targets)
                loss = [ criterion( output.view_as( target ), target ) for (criterion, output, target) in zip( self.criterions, outputs, targets ) ]
            elif isinstance( outputs, torch.Tensor ): # Multi-Loss   
                loss = [ criterion( outputs.view_as( target ), target ) for (criterion, target) in zip( self.criterions, targets ) ]
        else: # simple training
            loss = [ self.criterions[0](outputs.view_as( targets ), targets) ]
        if self.weights is not None:
            assert len(self.weights)==len(loss)
            loss = [ l*w for (l, w) in zip( loss, self.weights ) ]
        return loss


class SbmTask(TaskBase):
    def __init__(self, criterions=[], tasks=[]):
        self.tasks = []
        for i, criterion in enumerate(criterions):
            Task = getattr(sys.modules[__name__], tasks[i]+'Task')(criterion = criterion)
            self.tasks.append(Task)

    def get_loss(self, student, teachers, inputs, split_size):
        loss = 0.
        for i, (teacher, task) in enumerate(zip(teachers, self.tasks)):
            targets = task.predict(teacher, inputs)['preds']
            outputs = torch.split(student(inputs), split_size, dim=1)[i]
            loss += task.criterion(outputs, targets)
        return {'loss': loss}

    def predict(self, student, inputs, split_size):
        joint_outputs = torch.split(student(inputs), split_size, dim=1)
        return {'preds': joint_outputs}