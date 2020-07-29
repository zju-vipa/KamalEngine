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

import abc

from . import depara
from captum.attr import InputXGradient
import torch
from kamal.core import hub
from kamal.vision import sync_transforms as sT

class TransMetric(abc.ABC):
    def __init__(self):
        pass

    def __call__(self, a, b) -> float:
        return 0

class DeparaMetric(TransMetric):
    def __init__(self, data, device):
        self.data = data
        self.device = device
        self._cache = {}
    
    def _get_transform(self, metadata):
        input_metadata = metadata['input']
        size = input_metadata['size']
        space = input_metadata['space']
        drange = input_metadata['range']
        normalize = input_metadata['normalize']
        print(metadata)
        if size==None:
            size=224
        if isinstance(size, (list, tuple)):
            size = size[-1]
        transform = [
            sT.Resize(size),
            sT.CenterCrop(size),
        ]
        if space=='bgr':
            transform.append(sT.FlipChannels())
        if list(drange)==[0, 1]:
            transform.append( sT.ToTensor() )
        elif list(drange)==[0, 255]:
            transform.append( sT.ToTensor(normalize=False, dtype=torch.float) )
        else:
            raise NotImplementedError
        if normalize is not None:
            transform.append(sT.Normalize( mean=normalize['mean'], std=normalize['std'] ))
        return sT.Compose(transform)

    def _get_attr_graph(self, n):
        transform = self._get_transform(n.metadata)
        data = torch.stack( [ transform( d ) for d in self.data ], dim=0 )
        return depara.get_attribution_graph(
            n.model,
            attribution_type=InputXGradient,
            with_noise=False,
            probe_data=data,
            device=self.device
        )

    def __call__(self, n1, n2):
        attrgraph_1 = self._cache.get(n1, None)
        attrgraph_2 = self._cache.get(n2, None)
        if attrgraph_1 is None:
            self._cache[n1] = attrgraph_1 = self._get_attr_graph(n1)
        if attrgraph_2 is None:
            self._cache[n2] = attrgraph_2 = self._get_attr_graph(n2)
        return depara.graph_similarity(attrgraph_1, attrgraph_2)

class AttrMapMetric(DeparaMetric):
    def _get_attr_map(self, n):
        transform = self._get_transform(n.metadata)
        data = torch.stack( [ transform( d ).to(self.device) for d in self.data ], dim=0 )
        return depara.attribution_map(
            n.model.to(self.device),
            attribution_type=InputXGradient,
            with_noise=False,
            probe_data=data,
        )

    def __call__(self, n1, n2):
        attrgraph_1 = self._cache.get(n1, None)
        attrgraph_2 = self._cache.get(n2, None)
        if attrgraph_1 is None:
            self._cache[n1] = attrgraph_1 = self._get_attr_map(n1)
        if attrgraph_2 is None:
            self._cache[n2] = attrgraph_2 = self._get_attr_map(n2)
        return depara.attr_map_distance(attrgraph_1, attrgraph_2)
    
