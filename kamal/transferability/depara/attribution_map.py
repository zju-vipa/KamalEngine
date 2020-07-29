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
from typing import Type, Callable
from captum.attr import Attribution
from captum.attr import NoiseTunnel


def with_norm(func: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, square: bool = False):
    x = func(x)
    x = torch.norm(x.flatten(1), dim=1, p=2)
    if square:
        x = torch.pow(x, 2)
    return x


def attribution_map(
    func: Callable[[torch.Tensor], torch.Tensor],
    attribution_type: Type,
    with_noise: bool,
    probe_data: torch.Tensor,
    norm_square: bool = False,
    **attribution_kwargs
) -> torch.Tensor:
    """
    Calculate attribution map with given attribution type(algorithm).
    Args:
        model: pytorch module
        attribution_type: attribution algorithm, e.g. IntegratedGradients, InputXGradient, ...
        with_noise: whether to add noise tunnel
        probe_data: input data to model
        device: torch.device("cuda: 0")
        attribution_kwargs: other kwargs for attribution method
    Return: attribution map
    """
    attribution: Attribution = attribution_type(lambda x: with_norm(func, x, norm_square))
    if with_noise:
        attribution = NoiseTunnel(attribution)
    attr_map = attribution.attribute(
        inputs=probe_data,
        target=None,
        **attribution_kwargs
    )
    return attr_map.detach()

def attr_map_distance(map_1: torch.Tensor, map_2: torch.Tensor):
    if map_1.shape != map_2.shape:
        map_1 = torch.nn.functional.interpolate( map_1, size=map_2.shape[-2:] )
    dist = torch.dist(map_1.flatten(1), map_2.flatten(1), p=2).mean()
    return dist.item()

def attr_map_similarity(map_1: torch.Tensor, map_2: torch.Tensor):
    assert(map_1.shape == map_2.shape)
    dist = torch.cosine_similarity(map_1.flatten(1), map_2.flatten(1)).mean()
    return dist.item()


if __name__ == "__main__":
    import captum

    def ff(x):
        return x ** 2

    m = attribution_map(
        ff,
        captum.attr.InputXGradient,
        with_noise=False,
        probe_data=torch.tensor([[1, 2, 3, 4]], dtype=torch.float, requires_grad=True),
        norm_square=True
    )
    print(m)
