from typing import Union

from numbers import Number
import torch
import torch.distributions as td


class Normal(td.Independent):
    """
    Normal distribution on the d-dimensional space defined by
    mean and std tensors of any shape.

    Example:
    >>> dist = stribor.Normal(0., 1.)
    >>> dist.log_prob(torch.Tensor([0]))
    tensor([-0.9189])
    >>> dist = stribor.Normal(torch.zeros(2), torch.ones(2))
    >>> dist.log_prob(torch.zeros(2, 2))
    tensor([-1.8379, -1.8379])

    Args:
        loc (float or tensor): Mean
        scale (float or tensor): Standard deviation
    """

    def __init__(self, loc, scale, **kwargs):
        self.loc = loc
        self.scale = scale
        rbd = 0 if isinstance(self.loc, float) else 1
        super().__init__(td.Normal(self.loc, self.scale, **kwargs),
            reinterpreted_batch_ndims=rbd)


class UnitNormal(Normal):
    """
    Unit `st.Normal`. Specify only the dimension.

    Example:
    >>> dist = stribor.UnitNormal(2)
    >>> dist.log_prob(torch.ones(1, 2))
    tensor([-2.8379])

    Args:
        dim (int): Dimension of data
    """

    def __init__(self, dim, **kwargs):
        self.dim = dim
        super().__init__(torch.zeros(self.dim), torch.ones(self.dim), **kwargs)


class MultivariateNormal(td.MultivariateNormal):
    """
    Wrapper for `torch.distributions.MultivariateNormal`.

    Args:
        loc (Tensor): mean of the distribution
        covariance_matrix (Tensor): positive-definite covariance matrix
        precision_matrix (Tensor): positive-definite precision matrix
        scale_tril (Tensor): lower-triangular factor of covariance, with positive-valued diagonal
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
