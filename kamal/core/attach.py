from typing import Sequence, Callable
from numbers import Number
from kamal.core import exceptions

class AttachTo(object):
    """ Attach task, metrics or visualizer to specified tensors
    """
    def __init__(self, attach_to=None):
        if attach_to is not None and not isinstance(attach_to, (Sequence, Number, str, Callable) ):
            raise exceptions.InvalidMapping
        self._attach_to = attach_to

    def __call__(self, *tensors):
        if self._attach_to is not None:
            if isinstance(self._attach_to, Callable):
                return self._attach_to( *tensors )
            if isinstance(self._attach_to, Sequence):
                _attach_to = self._attach_to
            else:
                _attach_to = [ self._attach_to for _ in range(len(tensors)) ]
            _attach_to = _attach_to[:len(tensors)]
            tensors = [ tensor[index] for (tensor, index) in zip( tensors, _attach_to ) ]
        if len(tensors)==1:
            tensors = tensors[0]
        return tensors

    def __repr__(self):
        rep = "AttachTo: %s"%(self._attach_to)