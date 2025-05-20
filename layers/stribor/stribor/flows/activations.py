from typing import Callable, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import constraints
from layers.stribor.stribor import ElementwiseTransform


class ELU(ElementwiseTransform):
    """
    Exponential linear unit.
    Adapted from Pyro (https://pyro.ai).
    """
    bijective = True
    domain: constraints.real
    codomain: constraints.positive

    def forward(self, x, **kwargs):
        return F.elu(x)

    def inverse(self, y, **kwargs):
        zero = torch.zeros_like(y)
        log_term = torch.log1p(y)
        x = torch.max(y, zero) + torch.min(log_term, zero)
        return x

    def log_det_jacobian(self, x, y=None):
        log_diag_jacobian = self.log_diag_jacobian(x, y)
        return log_diag_jacobian.sum(-1, keepdim=True)

    def jacobian(self, x, y, **kwargs):
        return torch.diag_embed(self.log_diag_jacobian(x, y).exp())

    def log_diag_jacobian(self, x, y, **kwargs):
        return -F.relu(-x)


class LeakyReLU(ElementwiseTransform):
    """
    Leaky ReLU.
    For `x >= 0` returns `x`, else returns `negative_slope * x`.
    Adapted from Pyro (https://pyro.ai).

    Args:
        negative_slope (float): Controls the angle of the negative slope. Default: 0.01
    """
    bijective = True
    domain: constraints.real
    codomain: constraints.real

    def __init__(self, negative_slope=0.01, **kwargs):
        super().__init__()
        assert negative_slope > 0, '`negative_slope` must be positive'
        self.negative_slope = negative_slope

    def _leaky_relu(self, x, negative_slope):
        zeros = torch.zeros_like(x)
        y = torch.max(zeros, x) + negative_slope * torch.min(zeros, x)
        return y

    def forward(self, x, **kwargs):
        return self._leaky_relu(x, self.negative_slope)

    def inverse(self, y, **kwargs):
        return self._leaky_relu(y, 1 / self.negative_slope)

    def log_det_jacobian(self, x, y=None):
        return self.log_diag_jacobian(x, y).sum(-1, keepdim=True)

    def jacobian(self, x, y, **kwargs):
        return torch.diag_embed(self.log_diag_jacobian(x, y).exp())

    def log_diag_jacobian(self, x, y, **kwargs):
        left = torch.zeros_like(x)
        right = torch.ones_like(x) * math.log(self.negative_slope)
        return torch.where(x >= 0.0, left, right)


class ContinuousActivation(ElementwiseTransform):
    """
    Continuous activation function.
    At t=0 is identity, as t grows acts more like the original activation.

    Args:
        activation (callable): An activation function (e.g, `torch.tanh`)
        temperature (float): How fast the activation becomes like the original
            as t increases. Higher temperature -> faster
        learnable (bool): Whether temperature is a learnable parameter
    """

    def __init__(self, activation, temperature=1.0, learnable=False):
        super().__init__()
        self.activation = activation
        self.temperature = nn.Parameter(torch.tensor(temperature),
            requires_grad=learnable)

    def forward(self, x, t, **kwargs):
        w = torch.tanh(self.temperature * t)
        return x * (1 - w) + self.activation(x) * w

    def inverse(self, y, **kwargs):
        raise NotImplementedError

    def log_diag_jacobian(self, x, y, **kwargs):
        raise NotImplementedError

    def log_det_jacobian(self, x, y, **kwargs):
        raise NotImplementedError


class ContinuousTanh(ElementwiseTransform):
    """
    Continuous activation that is the solution to `dx(t)/dt = tanh(x(t))`.

    Args:
        log_time (bool): Whether to use time in log domain.
    """

    def __init__(self, log_time=False):
        super().__init__()
        self.log_time = log_time

    def forward(self, x, t, reverse=False, **kwargs):
        if self.log_time:
            t = torch.log1p(t)
        if reverse:
            t = -t
        return torch.asinh(t.exp() * torch.sinh(x))

    def inverse(self, y, t, **kwargs):
        return self.forward(y, t, True)

    def log_diag_jacobian(self, x, y, t, **kwargs):
        if self.log_time:
            t = torch.log1p(t)
        t = t.exp()
        diag_jac = t * torch.cosh(x) / torch.sqrt(torch.square(t * torch.
            sinh(x)) + 1)
        return diag_jac.log()

    def log_det_jacobian(self, x, y, t, **kwargs):
        log_diag_jac = self.log_diag_jacobian(x, y, t)
        return log_diag_jac.sum(-1, keepdim=True)
