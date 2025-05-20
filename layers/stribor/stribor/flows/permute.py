from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.stribor.stribor import ElementwiseTransform


class Flip(ElementwiseTransform):
    """
    Flips the order of indices along selected dims.

    Example:
    >>> f = stribor.Flip()
    >>> x = torch.tensor([[1, 2], [3, 4]])
    >>> f(x)[0]
    tensor([[2, 1],
            [4, 3]])
    >>> f = stribor.Flip([0, 1])
    >>> f(x)[0]
    tensor([[4, 3],
            [2, 1]])

    Args:
        dims (List[int]): Dimensions along which to flip the order of values.
            Default: [-1]
    """

    def __init__(self, dims=[-1]):
        super().__init__()
        self.dims = dims

    def forward(self, x, **kwargs):
        return torch.flip(x, self.dims)

    def inverse(self, y, **kwargs):
        return torch.flip(y, self.dims)

    def log_det_jacobian(self, x, y, **kwargs):
        return torch.zeros_like(x[..., :1])

    def log_diag_jacobian(self, x, y, **kwargs):
        return torch.eye(x.shape[-1]).flip(self.dims).diag().log().expand_as(x)


class Permute(ElementwiseTransform):
    """
    Permutes indices along the last dimension.

    Example:
    >>> torch.manual_seed(123)
    >>> f = stribor.Permute(3)
    >>> f(torch.tensor([1, 2, 3]))
    (tensor([2, 3, 1]), tensor([0, 0, 0]))
    >>> f.inverse(torch.tensor(tensor([2, 3, 1])))
    (tensor([1, 2, 3]), tensor([0, 0, 0]))

    Args:
        dim (int): Dimension of data
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.permutation = torch.randperm(dim)
        self.inverse_permutation = torch.empty(dim).long()
        self.inverse_permutation[self.permutation] = torch.arange(dim)

    def forward(self, x, **kwargs):
        y = x[..., self.permutation]
        return y

    def inverse(self, y, **kwargs):
        x = y[..., self.inverse_permutation]
        return x

    def log_det_jacobian(self, x, y, **kwargs):
        return torch.zeros_like(x[..., :1])

    def log_diag_jacobian(self, x, y, **kwargs):
        return torch.eye(self.dim)[self.permutation].diag().log().expand_as(x)
