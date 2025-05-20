from typing import Optional, Tuple

import torch
import torch.nn as nn
import layers.stribor.stribor as st
from layers.stribor.stribor import ElementwiseTransform


class Spline(ElementwiseTransform):
    """
    Spline flow (https://arxiv.org/abs/1906.04032).
    Elementwise transformation of input vector using spline functions
    defined on the interval (lower,upper).

    Code adapted from https://github.com/bayesiains/nsf

    Example:
    >>> torch.manual_seed(123)
    >>> dim, n_bins, latent_dim = 2, 5, 50
    >>> param_size = dim * (3 * n_bins - 1)
    >>> f = stribor.Spline(dim, n_bins, latent_net=stribor.net.MLP(latent_dim, [32], param_size))
    >>> f(torch.rand(1, dim), latent=torch.rand(1, latent_dim))
    (tensor([[0.0063, 0.8072]], tensor([[ 0.0348, -0.3526]])

    Args:
        dim (int): Dimension of data
        n_bins (int): Number of bins/spline knots.
        latent_net (nn.Module, optional): Neural network that takes `latent`
            and outputs transformation parameters of size `dim * (2 * n_bins + 2)`
            for cubic and `dim * (3 * n_bins - 1)` for quadratic spline. Default: None
        lower (float, optional): Lower bound of the transformation domain. Default: 0
        upper (float, optional): Upper bound of the transformation domain. Default: 1
        spline_type (str, optional): Which spline function to use.
            Options: `quadratic`, `cubic`. Default: cubic
        latent_net: Dimension of the input latent vector
    """

    def __init__(self, dim, n_bins, latent_net=None, lower=0, upper=1,
        spline_type='cubic', **kwargs):
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.dim = dim
        self.n_bins = n_bins
        self.latent_net = latent_net
        if spline_type == 'quadratic':
            self.spline = st.util.unconstrained_rational_quadratic_spline
            self.derivative_dim = n_bins - 1
        elif spline_type == 'cubic':
            self.spline = st.util.unconstrained_cubic_spline
            self.derivative_dim = 2
        else:
            raise ValueError(
                'spline_type must be either `quadratic` or `cubic`')
        if self.latent_net is None:
            self.width = nn.Parameter(torch.empty(self.dim, n_bins))
            self.height = nn.Parameter(torch.empty(self.dim, n_bins))
            self.derivative = nn.Parameter(torch.empty(self.dim, self.
                derivative_dim))
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.width)
        nn.init.xavier_uniform_(self.height)
        nn.init.xavier_uniform_(self.derivative)

    def _get_params(self, latent=None):
        if latent is None:
            return self.width, self.height, self.derivative
        else:
            params = self.latent_net(latent)
            params = params.view(*params.shape[:-1], self.dim, self.n_bins *
                2 + self.derivative_dim)
            width = params[..., :self.n_bins]
            height = params[..., self.n_bins:2 * self.n_bins]
            derivative = params[..., 2 * self.n_bins:]
            return width, height, derivative

    def forward(self, x, latent=None, **kwargs):
        y, _ = self.forward_and_log_diag_jacobian(x, latent)
        return y

    def inverse(self, y, latent=None, **kwargs):
        x, _ = self.inverse_and_log_diag_jacobian(y, latent)
        return x

    def forward_and_log_diag_jacobian(self, x, latent=None, *, reverse=
        False, **kwargs):
        w, h, d = self._get_params(latent)
        return self.spline(x, w, h, d, inverse=reverse, lower=self.lower,
            upper=self.upper)

    def inverse_and_log_diag_jacobian(self, y, latent=None, **kwargs):
        x, log_diag_jacobian = self.forward_and_log_diag_jacobian(y, latent,
            reverse=True)
        return x, log_diag_jacobian

    def forward_and_log_det_jacobian(self, x, latent=None, **kwargs):
        y, log_diag_jacobian = self.forward_and_log_diag_jacobian(x, latent)
        return y, log_diag_jacobian.sum(-1, keepdim=True)

    def inverse_and_log_det_jacobian(self, y, latent=None, **kwargs):
        x, log_diag_jacobian = self.forward_and_log_diag_jacobian(y, latent,
            reverse=True)
        return x, log_diag_jacobian.sum(-1, keepdim=True)

    def log_det_jacobian(self, x, y, latent=None, **kwargs):
        _, log_det_jacobian = self.forward_and_log_det_jacobian(x, latent)
        return log_det_jacobian

    def log_diag_jacobian(self, x, y, latent=None, **kwargs):
        _, log_diag_jacobian = self.forward_and_log_diag_jacobian(x, latent)
        return log_diag_jacobian
