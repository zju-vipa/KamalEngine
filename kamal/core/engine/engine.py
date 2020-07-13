import torch
import torch.nn as nn
import abc, math, weakref, typing, time
from typing import Any, Callable, Optional, Sequence
import numpy as np 

from kamal.core.engine.events import DefaultEvents
from kamal.core import tasks
from kamal.utils import set_mode, move_to_device, get_logger
from collections import defaultdict

import numbers
from enum import Enum
import contextlib

class Event(object):
    def __init__(self, value: str, event_trigger: Optional[Callable]=None ):
        if event_trigger is None:
            event_trigger =  Event.default_trigger
        self._trigger = event_trigger
        self._name_ = self._value_ = value

    @property
    def trigger(self):
        return self._trigger

    @property
    def name(self):
        return self._name_

    @property
    def value(self):
        return self._value_

    @staticmethod
    def default_trigger(engine):
        return True

    @staticmethod
    def once_trigger():
        is_triggered = False
        def wrapper(engine):
            if is_triggered:
                return False
            is_triggered=True
            return True
        return wrapper

    @staticmethod
    def every_trigger(every: int):
        def wrapper(engine):
            return every>0 and (engine.state.iter % every)==0
        return wrapper

    def __call__(self, every: Optional[int]=None, once: Optional[bool]=None, event_trigger: Optional[Callable]=None ):
        if every is not None:
            return Event(self.value, event_trigger=Event.every_trigger(every) )
        if once is not None:
            return Event(self.value, event_trigger=Event.once_trigger() )
        if event_trigger is not None:
            return Event( self.value, event_trigger=event_trigger )
        return Event()

    def __hash__(self):
        return hash(self._name_)
    
    def __eq__(self, other):
        if hasattr(other, 'value'):
            return self.value==other.value
        else:
            return

class DefaultEvents(Event, Enum):
    BEFORE_RUN = "before_train"
    AFTER_RUN = "after_train"

    BEFORE_EPOCH = "before_epoch"
    AFTER_EPOCH = "after_epoch"

    BEFORE_STEP = "before_step"
    AFTER_STEP = "after_step"

    BEFORE_GET_BATCH = "before_get_batch"
    AFTER_GET_BATCH = "after_get_batch"

    
class State(object):
    def __init__(self):
        self.iter = 0
        self.max_iter = None
        self.epoch_length = None
        self.dataloader = None
        self.seed = None

        self.metrics=dict()
        self.batch=None

    @property
    def current_epoch(self):
        if self.epoch_length is not None:
            return self.iter // self.epoch_length
        return None

    @property
    def max_epoch(self):
        if self.epoch_length is not None:
            return self.max_iter // self.epoch_length
        return None

    @property
    def current_batch_index(self):
        if self.epoch_length is not None:
            return self.iter % self.epoch_length
        return None

    @property
    def max_batch_index(self):
        return self.epoch_length

    def __repr__(self):
        rep = "State:\n"
        for attr, value in self.__dict__.items():
            if not isinstance(value, (numbers.Number, str, dict)):
                value = type(value)
            rep += "\t{}: {}\n".format(attr, value)
        return rep

class Engine(abc.ABC):
    def __init__(self, logger=None, tb_writer=None):
        self._logger = logger if logger else get_logger(name='kamal', color=True)
        self._tb_writer = tb_writer
        self._callbacks = defaultdict(list)
        self._allowed_events = [ *DefaultEvents ]
        self._state = State()
    
    def reset(self):
        self._state = State()

    def run(self, step_fn: Callable, dataloader, max_iter, start_iter=0, epoch_length=None):
        self.state.iter = self._state.start_iter = start_iter
        self.state.max_iter = max_iter
        self.state.epoch_length = epoch_length if epoch_length else len(dataloader)
        self.state.dataloader = dataloader
        self.state.dataloader_iter = iter(dataloader)
        self.state.step_fn = step_fn

        self.trigger_events(DefaultEvents.BEFORE_RUN)
        for self.state.iter in range( start_iter, max_iter ):
            if self.state.epoch_length!=None and \
                 self.state.iter%self.state.epoch_length==0: # Epoch Start
                    self.trigger_events(DefaultEvents.BEFORE_EPOCH)
            self.trigger_events(DefaultEvents.BEFORE_STEP)
            self.state.batch = self._get_batch()
            step_output = step_fn(self, self.state.batch)
            if isinstance(step_output, dict):
                self.state.metrics.update(step_output)
            self.trigger_events(DefaultEvents.AFTER_STEP)        
            if self.state.epoch_length!=None and \
                 (self.state.iter+1)%self.state.epoch_length==0: # Epoch End
                    self.trigger_events(DefaultEvents.AFTER_EPOCH)
        self.trigger_events(DefaultEvents.AFTER_RUN)

    def _get_batch(self):
        try:
            batch = next( self.state.dataloader_iter )
        except StopIteration:
            self.state.dataloader_iter = iter(self.state.dataloader) # reset iterator
            batch = next( self.state.dataloader_iter )
        return batch

    @property
    def state(self):
        return self._state

    @property
    def logger(self):
        return self._logger

    @property
    def tb_writer(self):
        return self._tb_writer

    def add_callback(self, event: Event, callbacks ):
        if not isinstance(callbacks, Sequence):
            callbacks = [callbacks]
        if event in self._allowed_events:
            for callback in callbacks:
                if callback not in self._callbacks[event]:
                    if event.trigger!=event.default_trigger:
                        callback = self._trigger_wrapper(self, event.trigger, callback )
                    self._callbacks[event].append( callback )

    @staticmethod
    def _trigger_wrapper(engine, trigger, callback):
        def wrapper(*args, **kwargs) -> Any:
            if trigger(engine):
                return callback(engine)
        return wrapper

    def trigger_events(self, *events):
        for e in events:
            if e in self._allowed_events:
                for callback in self._callbacks[e]:
                    callback(self)

    def register_events(self, *events):
        for e in events:
            if e not in self._allowed_events:
                self._allowed_events.apped( e )

    @contextlib.contextmanager
    def save_current_callbacks(self):
        temp = self._callbacks
        self._callbacks = defaultdict(list)
        yield
        self._callbacks = temp