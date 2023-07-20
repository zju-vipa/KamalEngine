import numpy as np
import ipdb
import torch
from kamal.core.metrics.stream_metrics import Metric
from typing import Callable

# 定义一个抽象基类，用于表示三个阈值内的相对误差百分比（1.25, 1.25^2, 1.25^3）的度量（Metric）
class PercentageOfRelativeErrors(Metric):
    def __init__(self, attach_to=None, threshold=1.25):
        super(PercentageOfRelativeErrors, self).__init__(attach_to=attach_to)
        self.threshold = threshold 
        self.reset()

    @torch.no_grad()
    def update(self, outputs, targets):
        outputs, targets = self._attach(outputs, targets)
        max_ratio = torch.max(outputs / targets, targets / outputs) # 计算预测值和真实值之间的最大比率
        percent = torch.mean((max_ratio < self.threshold).float(), dim=[1, 2, 3]) # 计算阈值内的相对误差百分比
        self._percent += torch.sum(percent)
        # ipdb.set_trace()
        self._cnt += outputs.shape[0]

    def get_results(self):
        return (self._percent / self._cnt).detach().cpu()
    
    def reset(self):
        self._percent = 0.0
        self._cnt = 0

# 定义一个子类，用于表示第一个阈值（1.25）内的相对误差百分比
class PercentageOfRelativeErrors_125(PercentageOfRelativeErrors):
    def __init__(self, attach_to=None):
        super(PercentageOfRelativeErrors_125, self).__init__(attach_to=attach_to, threshold=1.25) 

# 定义一个子类，用于表示第二个阈值（1.25^2）内的相对误差百分比
class PercentageOfRelativeErrors_1252(PercentageOfRelativeErrors):
    def __init__(self, attach_to=None):
        super(PercentageOfRelativeErrors_1252, self).__init__(attach_to=attach_to, threshold=1.25 ** 2) 

# 定义一个子类，用于表示第三个阈值（1.25^3）内的相对误差百分比
class PercentageOfRelativeErrors_1253(PercentageOfRelativeErrors):
    def __init__(self, attach_to=None):
        super(PercentageOfRelativeErrors_1253, self).__init__(attach_to=attach_to, threshold=1.25 ** 3) # 调用父类的__init__方法，并传入第三个阈值