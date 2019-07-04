import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTask(nn.Module):
    """
    MultiTask Learning
    """
    def __init__(self, student, teachers):
        super(MultiTask, self).__init__()

        if not isinstance(teachers, nn.ModuleList):
            teachers = nn.ModuleList(teachers)

        self.student = student
        self.teachers = teachers

    def fit(self, train_loader, **kargs):
        raise NotImplementedError
