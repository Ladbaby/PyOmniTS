from abc import ABCMeta, abstractmethod
from typing import List, Optional, Tuple, Union

import torch.nn as nn
from torch.distributions import Distribution


class Transform(nn.Module, metaclass=ABCMeta):

    @abstractmethod
    def forward(self, x, **kwargs):
        pass

    @abstractmethod
    def inverse(self, y, **kwargs):
        pass

    @abstractmethod
    def log_det_jacobian(self, x, y, **kwargs):
        pass

    def jacobian(self, x, y, **kwargs):
        raise NotImplementedError

    def forward_and_log_det_jacobian(self, x, **kwargs):
        y = self.forward(x, **kwargs)
        log_det_jacobian = self.log_det_jacobian(x, y, **kwargs)
        return y, log_det_jacobian

    def inverse_and_log_det_jacobian(self, y, **kwargs):
        x = self.inverse(y, **kwargs)
        log_det_jacobian = self.log_det_jacobian(x, y, **kwargs)
        return x, -log_det_jacobian


class ElementwiseTransform(Transform):

    @abstractmethod
    def log_diag_jacobian(self, x, y, **kwargs):
        pass

    def forward_and_log_diag_jacobian(self, x, **kwargs):
        y = self.forward(x, **kwargs)
        log_diag_jacobian = self.log_diag_jacobian(x, y, **kwargs)
        return y, log_diag_jacobian

    def inverse_and_log_diag_jacobian(self, y, **kwargs):
        x = self.inverse(y, **kwargs)
        log_diag_jacobian = self.log_diag_jacobian(x, y, **kwargs)
        return x, -log_diag_jacobian


class NormalizingFlow(Transform):
    """
    Normalizing flow for density estimation and efficient sampling.

    Example:
    >>> import layers.stribor.stribor as st
    >>> torch.manual_seed(123)
    >>> dim = 2
    >>> f = st.NormalizingFlow(st.UnitNormal(dim), [st.Affine(dim)])
    >>> f.log_prob(torch.randn(3, 2))
    tensor([[-1.7560], [-1.7434], [-2.1792]])
    >>> f.sample(2)
    tensor([[-0.5204,  0.4196]])

    Args:
        base_dist (torch.distributions.Distribution): Base distribution
        transforms (Transform): List of invertible transformations
    """

    def __init__(self, base_dist, transforms):
        super().__init__()
        self.base_dist = base_dist
        self.transforms = nn.ModuleList(transforms)

    def forward(self, x, **kwargs):
        for f in self.transforms:
            x = f(x, **kwargs)
        return x

    def inverse(self, y, **kwargs):
        for f in reversed(self.transforms):
            y = f.inverse(y, **kwargs)
        return y

    def forward_and_log_det_jacobian(self, x, **kwargs):
        log_det_jac = 0
        for f in self.transforms:
            x, ldj = f.forward_and_log_det_jacobian(x, **kwargs)
            log_det_jac += ldj
        return x, log_det_jac

    def inverse_and_log_det_jacobian(self, y, **kwargs):
        log_det_jac = 0
        for f in reversed(self.transforms):
            y, ldj = f.inverse_and_log_det_jacobian(y, **kwargs)
            log_det_jac += ldj
        return y, log_det_jac

    def log_prob(self, y, **kwargs):
        x, log_det_jac = self.inverse_and_log_det_jacobian(y, **kwargs)
        log_prob = self.base_dist.log_prob(x).unsqueeze(-1) + log_det_jac
        return log_prob

    def sample(self, num_samples, *, rsample: bool=False, **kwargs):
        if isinstance(num_samples, int):
            num_samples = num_samples,
        if rsample:
            x = self.base_dist.rsample(num_samples)
        else:
            x = self.base_dist.sample(num_samples)
        x = self.forward(x, **kwargs)
        return x

    def rsample(self, num_samples, **kwargs):
        return self.sample(num_samples, **kwargs)

    def log_det_jacobian(self, x, y, **kwargs):
        _, log_det_jacobian = self.forward_and_log_det_jacobian(x, **kwargs)
        return log_det_jacobian


class NeuralFlow(nn.Module):
    """
    Neural flow model.
    https://arxiv.org/abs/2110.13040

    Example:
    >>> import layers.stribor.stribor as st
    >>>

    Args:
        transforms (Transform): List of invertible transformations
            that satisfy initial condition F(x, t=0)=x.
    """

    def __init__(self, transforms):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, x, t, t0=None, **kwargs):
        if t0 is not None:
            for transform in reversed(self.transforms):
                x = transform.inverse(x, t=t0, **kwargs)
        for transform in self.transforms:
            x = transform(x, t=t, **kwargs)
        return x
