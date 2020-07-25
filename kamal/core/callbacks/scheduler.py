from .base import Callback
from typing import Sequence

class LRSchedulerCallback(Callback):
    r""" LR scheduler callback
    """
    def __init__(self, schedulers=None):
        super(LRSchedulerCallback, self).__init__()
        if not isinstance(schedulers, Sequence):
            schedulers = ( schedulers, )
        self._schedulers = schedulers

    def __call__(self, trainer):
        if self._schedulers is None:
            return
        for sched in self._schedulers:
            sched.step()