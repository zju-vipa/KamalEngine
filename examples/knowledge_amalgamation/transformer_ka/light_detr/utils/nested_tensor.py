from typing import Optional, List

import torch
from torch import Tensor, BoolTensor
import torchvision.transforms.functional as TF


__all__ = [
    "NestedTensor",
    "pad_to_nested_tensor",
    "padding_collate_fn",
    "pretrain_padding_collate_fn"
]


class NestedTensor:
    """
    Combined feature with mask
    """
    def __init__(self, tensors: Tensor, mask: Optional[BoolTensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device: torch.device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def cuda(self):
        return self.to(torch.device("cuda"))

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return "Tensor:\n{}\nMask:\n{}".format(self.tensors, self.mask)


def pad_to_nested_tensor(tensor_list: List[Tensor]):
    """
    Padding 2D tensors to its max w and h with mask
    """
    device = tensor_list[0].device
    all_shape: List[torch.Size] = list()
    for t in tensor_list:
        assert t.ndim == 3, "All tensors must have shape [c, h_i, w_i]"
        assert device == t.device, "All tensors must be on the same device"
        all_shape.append(t.shape)

    all_shape: Tensor = torch.as_tensor(all_shape)
    assert torch.all(all_shape[:, 0] == all_shape[0, 0]), "All tensors must have the same channel size"
    max_size, _ = all_shape[:, 1:].max(dim=0)
    max_h, max_w = max_size

    tensors: List[Tensor] = list()
    masks: List[BoolTensor] = list()
    for t in tensor_list:
        shape = t.shape[1:]
        t = TF.pad(t, [0, 0, max_w - shape[1], max_h - shape[0]])
        tensors.append(t)

        mask = torch.ones_like(t[0], dtype=torch.bool)
        mask[:shape[0], :shape[1]] = False
        masks.append(mask)
    tensors = torch.stack(tensors)
    masks = torch.stack(masks)
    return NestedTensor(tensors, masks)


def padding_collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = pad_to_nested_tensor(batch[0])
    return tuple(batch)


def pretrain_padding_collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = pad_to_nested_tensor(batch[0])
    batch[1] = torch.stack(batch[1])
    return tuple(batch)
