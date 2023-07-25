import copy
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from . import seq_dropout_functional as F


class SeqDropoutBase(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.permute = None

    def _record_permute(self, permute: torch.Tensor):
        self.permute = permute.clone()

    def set_info(self, info: Dict[str, Any]):
        self.info = info

    def forward(
        self,
        permute: torch.Tensor,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.BoolTensor] = None,
        pos: Optional[torch.Tensor] = None
    ) -> Dict[str, Optional[torch.Tensor]]:
        # record permute tensor
        self._record_permute(permute)

        src = F.apply_batch_permute(src, permute, perm_dim=0, batch_dim=1)
        if src_mask is not None:
            raise NotImplementedError("`src_mask` is not supported")
        if src_key_padding_mask is not None:
            src_key_padding_mask = F.apply_batch_permute(src_key_padding_mask, permute, perm_dim=-1, batch_dim=0)
            src_len = src.shape[0]
            if torch.any(src_key_padding_mask.sum(dim=-1) == src_len):
                err = "`src_key_padding_mask` have row(s) with all `True`, which will lead to `Nan` in multihead attention"
                raise RuntimeError(err)
        if pos is not None:
            pos = F.apply_batch_permute(pos, permute, perm_dim=0, batch_dim=1)

        ret = {
            "src": src,
            "src_mask": src_mask,
            "src_key_padding_mask": src_key_padding_mask,
            "pos": pos
        }
        return ret


class SeqIdentity(SeqDropoutBase):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.BoolTensor] = None,
        pos: Optional[torch.Tensor] = None
    ) -> Dict[str, Optional[torch.Tensor]]:
        self.permute = F.get_identity_permute(src.shape[0], device=src.device)
        ret = {
            "src": src,
            "src_mask": src_mask,
            "src_key_padding_mask": src_key_padding_mask,
            "pos": pos
        }
        return ret


class SeqEqualDropout(SeqDropoutBase):
    """
    Drop input sequence to 1 / n_parts, e.g.
        x1 x2 x3 x4 | y1 y2 y3 y4  ==> x1 x3 | y2 y4
    """
    def __init__(self, num_parts: int, **kwargs):
        super().__init__()
        self.num_parts = num_parts

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        src_key_padding_mask: Optional[torch.BoolTensor],
        pos: Optional[torch.Tensor]
    ) -> Dict[str, Optional[torch.Tensor]]:
        permute = F.get_equal_dropout_permute(src.shape[0], self.num_parts, device=src.device)
        return super().forward(permute=permute, src=src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask, pos=pos)


class SeqEqualDropout_v2(SeqDropoutBase):
    """
    Drop input sequence to arbitrary percent.
    """
    def __init__(self, keep_percent: float = 0.5, **kwargs):
        super().__init__()
        self.keep_percent = keep_percent

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        src_key_padding_mask: Optional[torch.BoolTensor],
        pos: Optional[torch.Tensor]
    ) -> Dict[str, Optional[torch.Tensor]]:
        seq_len = src.shape[0]
        num_keep = round(seq_len * self.keep_percent)
        permute = F.get_equal_dropout_permute_v2(seq_len, num_keep, device=src.device)
        return super().forward(permute=permute, src=src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask, pos=pos)


class SeqRandomDropout(SeqDropoutBase):
    """
    Random permute input sequence
    """
    def __init__(self, keep_percent: float = 0.5, **kwargs):
        super().__init__()
        self.keep_percent = keep_percent

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        src_key_padding_mask: Optional[torch.BoolTensor],
        pos: Optional[torch.Tensor]
    ) -> Dict[str, Optional[torch.Tensor]]:
        seq_len = src.shape[0]
        num_keep = round(seq_len * self.keep_percent)
        permute = F.get_random_dropout_permute(
            seq_len,
            batch_size=src.shape[1],
            n_keep=num_keep,
            device=src.device
        )
        return super().forward(permute=permute, src=src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask, pos=pos)


class SeqSimDropout(SeqDropoutBase):
    """
    Random permute input sequence
    """
    def __init__(
        self,
        num_parts: int,
        merge_by_epoch: bool = False,
        norm: bool = False,
        **kwargs
    ):
        super().__init__()
        self.num_parts = num_parts
        self.merge_by_epoch = merge_by_epoch
        self.norm = norm

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        src_key_padding_mask: Optional[torch.BoolTensor],
        pos: Optional[torch.Tensor],
    ) -> Dict[str, Optional[torch.Tensor]]:
        if self.training:
            seq = self.info["feat_t"]
            seq = torch.cat(seq, dim=0)
            # merge teacher and student seq
            if self.merge_by_epoch:
                ep = self.info["epoch"]
                total_ep = self.info["total_epoch"]
                p = ep / total_ep
                seq = p * src + (1 - p) * seq
        else:
            seq = src

        permute = F.get_sim_dropout_permute(
            seq,
            self.num_parts,
            norm=self.norm
        )
        return super().forward(permute=permute, src=src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask, pos=pos)


__REGISTERED_SEQ_DROPOUT__ = {
    "identity": SeqIdentity,
    "equal_drop": SeqEqualDropout,
    "equal_drop_v2": SeqEqualDropout_v2,
    "random_drop": SeqRandomDropout,
    "sim_drop": SeqSimDropout
}


def build_seq_dropout(model_cfg: Dict[str, Any]) -> SeqDropoutBase:
    if "seq_drop" not in model_cfg:
        return
    seq_dropout_cfg = model_cfg["seq_drop"]
    seq_dropout_cfg["num_parts"] = model_cfg["detr"].get("num_proj", 1)
    seq_dropout_cfg = copy.deepcopy(seq_dropout_cfg)
    name = seq_dropout_cfg.pop("name")
    return __REGISTERED_SEQ_DROPOUT__[name](**seq_dropout_cfg)
