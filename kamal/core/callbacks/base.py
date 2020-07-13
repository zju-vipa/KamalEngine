import abc

class Callback(abc.ABC):
    r""" Base Class for Callbacks
    """
    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(self, engine):
        pass