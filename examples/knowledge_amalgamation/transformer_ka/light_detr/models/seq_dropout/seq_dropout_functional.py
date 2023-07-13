import torch

from .utils import get_dim_permute, get_seq_similarity


def get_identity_permute(seq_len: int, device: torch.device) -> torch.LongTensor:
    return torch.arange(0, seq_len, device=device)


def get_equal_dropout_permute(seq_len: int, num_parts: int, device: torch.device) -> torch.LongTensor:
    assert seq_len % num_parts == 0, "input sequence must be divided exactly by num_parts"
    m = num_parts
    n = seq_len // m
    ii = torch.arange(0, m).repeat(n, 1).T
    jj = torch.arange(0, n).repeat(m, 1)
    mask = jj % m == ii
    permute = get_identity_permute(seq_len, device)[mask.flatten()]
    return permute


def get_equal_dropout_permute_v2(
    seq_len: int,
    n_keep: int,
    device: torch.device
) -> torch.LongTensor:
    select = torch.arange(0, n_keep, dtype=torch.float, device=device)
    select *= seq_len / n_keep
    select = select.round().to(torch.int64)
    return select


def get_random_dropout_permute(
    seq_len: int,
    batch_size: int,
    n_keep: int,
    device: torch.device,
    sorted_index: bool = True,
) -> torch.LongTensor:
    permute = list()
    for _ in range(batch_size):
        permute.append(torch.randperm(seq_len, device=device))
    permute = torch.stack(permute)
    permute = permute[:, :n_keep]
    if sorted_index:
        permute, _ = permute.sort(dim=1)
    return permute


def get_sim_dropout_permute(
    seq: torch.Tensor,
    num_parts: int,
    norm: bool = False,
    sort: bool = True
) -> torch.LongTensor:
    """
    Get compress permutation to decrease total self-similarity, so that out sequence has shape [N_new, bs, dim]
    Args:
        seq: transformer sequence with shape [N, bs, dim]
        n_keep: n_keep * num_parts = N_new
        num_parts: number of parts (teachers)
    """
    seq_len = seq.shape[0]
    assert seq_len % num_parts == 0
    n = seq_len // num_parts

    # sequence similarity [bs, N]
    seq_sim = get_seq_similarity(seq, norm)
    # [bs, n_parts, n]
    seq_sim = seq_sim.unflatten(1, (num_parts, n))
    keep_by_teacher = torch.argmin(seq_sim, dim=1)
    bias = torch.arange(0, n, device=seq.device)
    keep = keep_by_teacher * n + bias
    if sort:
        keep, _ = keep.sort(dim=1)
    return keep


def _apply_batch_permute(tensor: torch.Tensor, permute: torch.LongTensor):
    """
    Args:
        tensor: [bs, N, ...]
        permute: [bs, N_new]
    """
    bs, n = tensor.shape[0:2]
    n_new = permute.shape[1]
    tensor = tensor.flatten(0, 1)
    step = torch.arange(0, bs, device=tensor.device) * n
    step = step.expand(n_new, -1).T
    permute = permute.flatten(0, 1) + step.flatten(0, 1)
    tensor = tensor.index_select(dim=0, index=permute)
    tensor = tensor.unflatten(0, sizes=(bs, n_new))
    return tensor


def apply_batch_permute(tensor: torch.Tensor, permute: torch.LongTensor, perm_dim: int = 0, batch_dim: int = -1):
    """
    Apply permutations to all batch sequence
    Args:
        seq: tensor with shape [..., N, ...] where N is `dim`
        permute: tensor with shape [N_new] or [bs, N_new]
    """
    # all batch shares the same permute
    if permute.dim() == 1:
        return tensor.index_select(dim=perm_dim, index=permute)
    # all batch shares different permute
    elif permute.dim() == 2:
        # permute to [bs, N, ..., 0, 1]
        dim = tensor.dim()
        perm_dim = perm_dim % dim
        batch_dim = batch_dim % dim
        flag = batch_dim != 0 or perm_dim != 1
        if flag:
            dim_perm, dim_perm_inv = get_dim_permute(dim, perm_dim, batch_dim)
            tensor = tensor.permute(*dim_perm)
        tensor = _apply_batch_permute(tensor, permute)
        if flag:
            tensor = tensor.permute(*dim_perm_inv)
        return tensor
    else:
        raise NotImplementedError

