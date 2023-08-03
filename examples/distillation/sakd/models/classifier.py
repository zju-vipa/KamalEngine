from __future__ import print_function

import torch.nn as nn
import torchvision.models as models
import torch

#########################################
# ===== Classifiers ===== #
#########################################

class LinearClassifier(nn.Module):

    def __init__(self, dim_in, n_label=10):
        super(LinearClassifier, self).__init__()

        self.net = nn.Linear(dim_in, n_label)
        self.n_label = n_label
        self.n = False

    def set_n_to_True(self):
        for i in range(self.n_label):
            with torch.no_grad():
                if i % 2 == 0:
                    self.net.bias[i] += 10
        self.net.bias = nn.Parameter(self.net.bias)

    def forward(self, x):
        return self.net(x)


class NonLinearClassifier(nn.Module):

    def __init__(self, dim_in, n_label=10, p=0.1):
        super(NonLinearClassifier, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(dim_in, 200),
            nn.Dropout(p=p),
            nn.BatchNorm1d(200),
            nn.ReLU(inplace=True),
            nn.Linear(200, n_label),
        )

    def forward(self, x):
        return self.net(x)
