# https://github.com/huawei-noah/Data-Efficient-Model-Compression
import torch.nn as nn

class LeNet5(nn.Module):

    def __init__(self, nc=1, num_classes=10):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(5, 5)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(6, 16, kernel_size=(5, 5)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(16, 120, kernel_size=(5, 5)),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes)
        )

    def forward(self, img, return_features=False):
        features = self.features( img ).view(-1, 120)
        output = self.fc( features )
        if return_features:
            return output, features
        return output
    

class LeNet5Half(nn.Module):

    def __init__(self, nc=1, num_classes=10):
        super(LeNet5Half, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=(5, 5)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(3, 8, kernel_size=(5, 5)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(8, 60, kernel_size=(5, 5)),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(60, 42),
            nn.ReLU(inplace=True),
            nn.Linear(42, num_classes)
        )

    def forward(self, img, return_features=False):
        features = self.features( img ).view(-1, 60)
        output = self.fc( features )
        if return_features:
            return output, features
        return output