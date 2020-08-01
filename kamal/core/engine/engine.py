# Copyright 2020 Zhejiang Lab. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================

import torch
import torch.nn as nn
import abc, math, weakref, typing, time
from typing import Any, Callable, Optional, Sequence
import numpy as np 

from kamal.core.engine.events import DefaultEvents, Event
from kamal.core import tasks
from kamal.utils import set_mode, move_to_device, get_logger
from collections import defaultdict

import numbers
import contextlib

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
        if not isinstance(batch, (list, tuple)):
            batch = [ batch, ] # no targets
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
        callbacks = [ RemovableCallback(self, event, c) for c in callbacks ]
        return ( callbacks[0] if len(callbacks)==1 else callbacks )

    def remove_callback(self, event, callback):
        for c in self._callbacks[event]:
            if c==callback:
                self._callbacks.remove( callback )
                return True
        return False

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

class RemovableCallback:
    def __init__(self, engine, event, callback):
        self._engine = weakref.ref(engine)
        self._callback = weakref.ref(callback)
        self._event = weakref.ref(event)
    
    @property
    def callback(self):
        return self._callback()

    def remove(self):
        engine = self._engine()
        callback = self._callback()
        event = self._event()
        return engine.remove_callback(event, callback)

        