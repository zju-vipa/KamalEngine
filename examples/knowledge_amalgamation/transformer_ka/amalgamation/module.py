from typing import Tuple

from torch import nn, Tensor


class ChannelNorm(nn.Module):
    def __init__(self, normalized_shape, dim: Tuple[int], elementwise_affine: bool = True):
        super().__init__()
        if isinstance(dim, int):
            dim = (dim,)
        self.dim = dim
        self.layer_norm = nn.LayerNorm(normalized_shape, elementwise_affine=elementwise_affine)

    def forward(self, x: Tensor) -> Tensor:
        x_mean = x.mean(dim=self.dim)
        x = self.layer_norm(x - x_mean)
        return x

