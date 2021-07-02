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

from typing import Callable, Optional
from enum import Enum

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
        """The name of the Enum member."""
        return self._name_

    @property
    def value(self):
        """The value of the Enum member."""
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

    def __call__(self, every: Optional[int]=None, once: Optional[bool]=None ):
        if every is not None:
            assert once is None
            return Event(self.value, event_trigger=Event.every_trigger(every) )
        if once is not None:
            return Event(self.value, event_trigger=Event.once_trigger() )
        return Event(self.value)

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

    