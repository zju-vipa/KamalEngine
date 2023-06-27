from typing import Dict, Any, Iterable
import copy

from torch.optim import *


OptimizerDict = dict(
    SGD=SGD,
    Adadelta=Adadelta,
    Adagrad=Adagrad,
    Adam=Adam,
    AdamW=AdamW,
    SparseAdam=SparseAdam,
    Adamax=Adamax,
    ASGD=ASGD,
    Rprop=Rprop,
    RMSprop=RMSprop,
    LBFGS=LBFGS
)


def get_optimizer(params: Iterable, optim_cfg: Dict[str, Any]) -> Optimizer:
    name = optim_cfg["name"]
    optimizer = OptimizerDict[name]

    kwargs = copy.deepcopy(optim_cfg)
    kwargs.pop("name")

    return optimizer(params=params, **kwargs)
