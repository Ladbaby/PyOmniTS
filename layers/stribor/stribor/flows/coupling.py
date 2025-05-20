from typing import Optional, Tuple

import torch
import torch.nn as nn
import layers.stribor.stribor as st
from layers.stribor.stribor import Transform, ElementwiseTransform


class Coupling(Transform):
    """
    Coupling transformation via elementwise transforms. If `dim = 1`, set `mask` to `'none'`.
    Splits data into 2 parts based on a mask. One part generates parameters of the transform
    that will transform the rest. Efficient computation in both directions.

    Example:
    >>> import layers.stribor.stribor as st
    >>> torch.manual_seed(123)
    >>> dim, n_bins, latent_dim = 2, 5, 50
    >>> f = st.Coupling(st.Affine(dim, st.net.MLP(dim + latent_dim, [64], 2 * dim)), mask='ordered_left_half')
    >>> f(torch.randn(1, 2), latent=torch.randn(1, latent_dim))
    tensor([[ 0.4974, -1.6288]], tensor([[0.0000, 0.3708]]
    >>> f = st.Coupling(st.Spline(dim, n_bins, st.net.MLP(dim, [64], dim * (3 * n_bins - 1))), mask='random_half')
    >>> f(torch.rand(1, dim))
    tensor([[0.9165, 0.7125]], tensor([[0.6281, -0.0000]]

    Args:
        transform (Transform): Elementwise transform with `latent_net` property.
            Latent network takes input of size `dim` and returns the parameters of the transform.
        mask (str): Mask name from `stribor.util.mask`.
            Options: `none`, `ordered_right_half` (right transforms left), `ordered_left_half`, `random_half`,
            `parity_even` (even indices transform odd), `parity_odd`.
        set_data (bool): Whether data has shape (..., N, dim) and we want to be permutation invariant. Default: False
    """

    def __init__(self, transform, mask, set_data=False, **kwargs):
        super().__init__()
        self.transform = transform
        self.mask_func = st.util.mask.get_mask(mask)
        self.set_data = set_data

    def _get_mask(self, x):
        if self.set_data:
            *rest, N, D = x.shape
            return self.mask_func(N).unsqueeze(-1).expand(*rest, N, D).to(x)
        else:
            return self.mask_func(x.shape[-1]).expand_as(x).to(x)

    def _get_conditioning(self, x, mask, latent=None):
        z = x * mask
        if x.shape[-1] == 1:
            z = z * 0
        if latent is not None:
            z = torch.cat([z, latent], -1)
        return z

    def forward(self, x, latent=None, reverse=False, **kwargs):
        mask = self._get_mask(x)
        z = self._get_conditioning(x, mask, latent)
        if reverse:
            y_ = self.transform.inverse(x, latent=z, **kwargs)
        else:
            y_ = self.transform(x, latent=z, **kwargs)
        y = y_ * (1 - mask) + x * mask
        return y

    def inverse(self, y, latent=None, **kwargs):
        return self.forward(y, latent, reverse=True)

    def log_det_jacobian(self, x, y=None, latent=None, **kwargs):
        mask = self._get_mask(x)
        z = self._get_conditioning(x, mask, latent)
        log_diag_jacobian = self.transform.log_diag_jacobian(x, y, latent=z,
            **kwargs)
        return (log_diag_jacobian * (1 - mask)).sum(-1, keepdim=True)


class ContinuousAffineCoupling(Transform):
    """
    Continuous affine coupling layer. If `dim = 1`, set `mask = 'none'`.
    Similar to `Coupling` but applies only an affine transformation
    which here depends on time `t` such that it's identity map at `t = 0`.

    Example:
    >>> import layers.stribor.stribor as st
    >>> torch.manual_seed(123)
    >>> dim = 2
    >>> f = st.ContinuousAffineCoupling(st.net.MLP(dim+1, [64], 2 * dim), st.net.TimeLinear(2 * dim), 'parity_odd')
    >>> f(torch.rand(1, 2), t=torch.rand(1, 1))
    (tensor([[0.8188, 0.4037]], tensor([[-0.0000, -0.1784]])

    Args:
        latent_net (nn.Module): Inputs concatenation of `x` and `t` (and optionally
            `latent`) and outputs affine transformation parameters (size `2 * dim`)
        time_net (stribor.net.time_net): Time embedding with the same output
            size as `latent_net`
        mask (str): Mask name from `stribor.util.mask`
            Options: `none`, `ordered_right_half` (right transforms left), `ordered_left_half`, `random_half`,
            `parity_even` (even indices transform odd), `parity_odd`.
        concatenate_time (bool): Whether to add time to input.
    """

    def __init__(self, latent_net, time_net, mask, concatenate_time=True,
        **kwargs):
        super().__init__()
        self.latent_net = latent_net
        self.mask_func = st.util.mask.get_mask(mask)
        self.time_net = time_net
        self.concatenate_time = concatenate_time

    def _get_mask(self, x):
        return self.mask_func(x.shape[-1]).expand_as(x).to(x)

    def _get_conditioning(self, x, t, mask, latent=None):
        z = x * mask
        if x.shape[-1] == 1:
            z = z * 0
        if latent is not None:
            z = torch.cat([z, latent], -1)
        if self.concatenate_time:
            z = torch.cat([z, t], -1)
        return z

    def forward(self, x, t, latent=None, **kwargs):
        y, _ = self.forward_and_log_det_jacobian(x, t, latent)
        return y

    def inverse(self, y, t, latent=None, **kwargs):
        x, _ = self.inverse_and_log_det_jacobian(y, t, latent)
        return x

    def log_det_jacobian(self, x, y, *, t, latent=None, **kwargs):
        _, log_det_jacobian = self.forward_and_log_det_jacobian(x, t, latent)
        return log_det_jacobian

    def forward_and_log_det_jacobian(self, x, t, latent=None, *, reverse:
        bool=False, **kwargs):
        mask = self._get_mask(x)
        z = self._get_conditioning(x, t, mask, latent)
        log_scale, shift = self.latent_net(z).chunk(2, dim=-1)
        t_log_scale, t_shift = self.time_net(t).chunk(2, dim=-1)
        if reverse:
            y = (x - shift * t_shift) * torch.exp(-log_scale * t_log_scale)
        else:
            y = x * torch.exp(log_scale * t_log_scale) + shift * t_shift
        log_diag_jacobian = log_scale * t_log_scale * (1 - mask)
        y = y * (1 - mask) + x * mask
        return y, log_diag_jacobian.sum(-1, keepdim=True)

    def inverse_and_log_det_jacobian(self, y, t, latent=None, **kwargs):
        x, log_det_jacobian = self.forward_and_log_det_jacobian(y, t,
            latent, reverse=True)
        return x, -log_det_jacobian
