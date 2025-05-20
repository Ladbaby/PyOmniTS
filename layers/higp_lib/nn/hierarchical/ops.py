import torch
from einops import repeat


def src_reduce(x, s):
    # aggregate the information from the lower level as a weighted mean
    w = 1 / s.sum(-2, keepdims=True)
    if s.dim() == 3:
        x_ = torch.einsum('b n k, b ... n f -> b ... k f', w * s, x)
    else:
        x_ = torch.einsum('n k, ... n f -> ... k f', w * s, x)
    return x_


def src_lift(x, s):
    # propagate the information from the higher level to the lower level
    if s.dim() == 3:
        x_ = torch.einsum('b n k, b ... k f -> b ... n f', s, x)
    else:
        x_ = torch.einsum('n k, ... k f -> ... n f', s, x)
    return x_


def src_connect(adj, s):
    return torch.matmul(torch.matmul(s.transpose(-2, -1), adj), s)


def compute_aggregation_matrix(selects, bottom_up=True):
    assert selects[0] is None
    selects = selects[1:]
    s = selects[0]
    C = [s]
    for i in range(1, len(selects)):
        # batched matmul if needed
        s = torch.matmul(s, selects[i])
        C.append(s)
    if bottom_up:
        C = C[::-1]
    C = torch.cat(C, dim=-1).transpose(-2, -1)
    return C


def build_three_level_hierarchy(x, select):
    xs = [x.mean(-2, keepdims=True), src_reduce(x, select), x]
    sizes = [xi.size(-2) for xi in xs]
    xs = torch.cat(xs, dim=-2)
    selects = [None,
               select,
               torch.ones(select.shape[-1], 1, device=select.device)]
    return xs, selects, sizes


def build_Q(C):
    C_ = C / C.sum(-1, keepdims=True)
    I_m = torch.eye(C.size(-2), device=C.device)
    if C.dim() == 3:
        I_m = repeat(I_m, 'n m -> b n m', b=C.size(0))
    Q = torch.cat([I_m, - C_], dim=-1)
    return Q
