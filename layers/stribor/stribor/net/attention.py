import layers.stribor.stribor as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(query, key, value, n_heads=1, mask_diagonal=False, mask=None):
    """
    Multihead attention layer with optional masking.
    "Attention Is All You Need" (https://arxiv.org/abs/1706.03762)

    Args:
        query (tensor): Query matrix (..., N, D)
        key (tensor): Key matrix (..., N, D)
        value (tensor): Value matrix (..., N, D)
        h_heads (int, optional): Number of attention heads, must divide D. Default: 1
        mask_diagonal (bool, optional): Whether to mask the diagonal in QK^T matrix,
            i.e., if self-interaction is disallowed. Default: False
        mask (tensor, optional): Mask with shape (..., N, 1), masked elements
            will be ignored. Default: None

    Returns:
        y (tensor): Result of multihead attention with shape (..., N, D)
    """
    *query_shape, _ = query.shape
    *value_shape, D = value.shape
    query = query.view(*query_shape, n_heads, D // n_heads).transpose(-2, -3)
    value = value.view(*value_shape, n_heads, D // n_heads).transpose(-2, -3)
    key = key.view(*value_shape, n_heads, D // n_heads).transpose(-2, -3)
    att = query @ key.transpose(-1, -2) * (1 / key.shape[-1]) ** 0.5
    if mask_diagonal:
        att.masked_fill_(torch.eye(att.shape[-1]).bool(), -np.inf)
    if mask is not None:
        att_mask = 1 - mask.transpose(-1, -2).unsqueeze(-2).repeat_interleave(
            att.shape[-2], dim=-2)
        att.masked_fill_(att_mask.bool(), -np.inf)
    att = st.util.safe_softmax(att, -1)
    y = att @ value
    y = y.transpose(-2, -3).reshape(*query_shape, -1)
    if mask is not None and y.shape[-2] == mask.shape[-2]:
        y = y * mask
    return y


class Attention(nn.Module):
    """
    Attention layer.

    Example:
    >>> net = stribor.net.Attention(3, [64, 64], 8, n_heads=4)
    >>> x = torch.randn(32, 10, 3)
    >>> net(x, x, x).shape
    torch.Size([32, 10, 8])

    Args:
        in_dim (int): Input size
        hidden_dims (List[int]): Hidden dimensions of embedding network, last dimmension
            is used as an embedding size for queries, keys and values
        out_dim (int): Output size
        h_heads (int, optional): Number of attention heads, must divide last element
            of `hidden_dims`. Default: 1
        mask_diagonal (bool, optional): Whether to mask the diagonal in similarity matrix,
            i.e., if self-interaction is disallowed. Default: False
    """

    def __init__(self, in_dim, hidden_dims, out_dim, n_heads=1,
        mask_diagonal=False, **kwargs):
        super().__init__()
        self.mask_diagonal = mask_diagonal
        self.n_heads = n_heads
        self.key = st.net.MLP(in_dim, hidden_dims[:-1], hidden_dims[-1])
        self.query = st.net.MLP(in_dim, hidden_dims[:-1], hidden_dims[-1])
        self.value = st.net.MLP(in_dim, hidden_dims[:-1], hidden_dims[-1])
        self.proj = nn.Linear(hidden_dims[-1], out_dim)

    def forward(self, query, key, value, mask=None, **kwargs):
        """
        Args:
            query (tensor): Queries with shape (..., N, in_dim)
            key (tensor): Keys with shape (..., N, in_dim)
            value (tensor): Values with shape (..., N, in_dim)
            mask (tensor, optional): Queries with shape (..., N, 1)

        Returns:
            y (tensor): Outputs with shape (..., N, out_dim)
        """
        y = attention(self.query(query), self.key(key), self.value(value),
            self.n_heads, self.mask_diagonal, mask)
        return self.proj(y)


class SelfAttention(Attention):
    """
    Self attention layer.
    Same as `st.net.Attention` but `forward` takes only `x` and `mask`.

    Args:
        Same as in `st.net.Attention`
    """

    def __init__(self, in_dim, hidden_dim, out_dim, n_heads=1,
        mask_diagonal=False, **kwargs):
        super().__init__(in_dim, hidden_dim, out_dim, n_heads, mask_diagonal)

    def forward(self, x, mask=None, **kwargs):
        """
        Args:
            x (tensor): Input, acts as query, key and value, with shape (..., N, in_dim)
            mask (tensor, optional): Queries with shape (..., N, 1)

        Returns:
            y (tensor): Outputs with shape (..., N, out_dim)
        """
        return super().forward(x, x, x, mask=mask)


class InducedSelfAttention(nn.Module):
    """
    Induced self attention that used inducing points to reduce the complexity
    of computing the attention for big sets.
    "Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks"
    (https://arxiv.org/abs/1810.00825)

    Args:
        in_dim (int): Input size
        hidden_dims (List[int]): Hidden dimensions of embedding network, last dimmension
            is used as an embedding size for queries, keys and values
        out_dim (int): Output size
        h_heads (int, optional): Number of attention heads, must divide last element
            of `hidden_dims`. Default: 1
        n_points (int, optional): Number of inducing point. Default: 32
    """

    def __init__(self, in_dim, hidden_dim, out_dim, n_heads=1, n_points=32,
        **kwargs):
        super().__init__()
        self.att1 = Attention(in_dim, hidden_dim, in_dim, n_heads)
        self.att2 = Attention(in_dim, hidden_dim, out_dim, n_heads)
        self.points = nn.Parameter(torch.empty(n_points, in_dim).uniform_(-
            1.0, 1.0))

    def forward(self, x, mask=None, **kwargs):
        h = self.points[(None,) * (len(x.shape) - 2)].repeat(*x.shape[:-2],
            1, 1)
        h = self.att1(h, x, x, mask=mask, **kwargs)
        y = self.att2(x * (1 if mask is None else mask), h, h, **kwargs)
        return y
