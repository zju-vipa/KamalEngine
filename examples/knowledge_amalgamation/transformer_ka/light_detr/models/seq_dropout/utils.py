from typing import Tuple

import torch
import torch.nn.functional as F


def interpolate_seq(seq: torch.Tensor, out_len: int, mode: str = "nearest"):
    """
    Interpolate a input sequence [N, bs, dim] to [out_len, bs, dim]
    """
    seq_len = seq.shape[1]
    if seq_len == out_len:
        return seq
    seq = seq.permute(1, 2, 0)
    seq = F.interpolate(seq, out_len, mode=mode)
    seq = seq.permute(2, 0, 1)
    return seq


def get_dim_permute(dim: int, perm_dim: int, batch_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Permute `batch_dim` to 0-dim and `perm_dim` to 1-dim

    Return:
        permute and inverse permute
    """
    assert perm_dim >= 0 and batch_dim >= 0

    origin_dim_p = torch.arange(0, dim)
    dim_perm = torch.zeros_like(origin_dim_p)
    mask = torch.ones_like(dim_perm, dtype=torch.bool)
    dim_perm[0] = batch_dim
    dim_perm[1] = perm_dim
    mask[[batch_dim, perm_dim]] = False
    dim_perm[2:] = origin_dim_p[mask]
    dim_perm_inv = torch.empty_like(dim_perm)
    dim_perm_inv[dim_perm] = torch.arange(dim)
    return dim_perm, dim_perm_inv


def get_sim_matrix(seq: torch.Tensor, norm: bool = False):
    """
    Get sequence token similarity value
    Args:
        seq: transformer sequence with shape [N, bs, dim]
    Return:
        token similarity matrix with shape [bs, N, N]
    """
    # [bs, N, dim]
    src_perm = seq.permute(1, 0, 2)
    # [bs, N, N]
    sim_matrix = torch.bmm(src_perm, src_perm.permute(0, 2, 1))
    if norm:
        # [bs, N]
        norm: torch.Tensor = torch.norm(src_perm, p=2, dim=-1, keepdim=True)
        norm = torch.bmm(norm, norm.permute(0, 2, 1)) + 1e-5
        sim_matrix /= norm
    return sim_matrix


def get_seq_similarity(seq: torch.Tensor, norm: bool = False) -> torch.LongTensor:
    """
    Get sequence token similarity value
    Args:
        seq: transformer sequence with shape [N, bs, dim]
    Return:
        token similarity with shape [bs, N]
    """
    N = seq.shape[0]
    # [bs, N, N]
    sim_matrix = get_sim_matrix(seq, norm)
    # [bs, N]
    sim = torch.sum(sim_matrix, dim=-1) / N
    return sim
