import torch
from torch import nn
from torch.nn import Module


class FeatureClassify(Module):
    def __init__(self, input_channels: int, num_classes: int):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(self.input_channels, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.global_avg_pool(x)
        x = x.squeeze()
        x = self.fc(x)
        return x


