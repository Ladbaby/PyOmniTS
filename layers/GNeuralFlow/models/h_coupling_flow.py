import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.GNeuralFlow.models.mask import get_mask
from layers.GNeuralFlow.models.mlp import MLP


class ContinuousAffineCoupling(nn.Module):
    """
    Continuous affine coupling layer. If `dim = 1`, set `mask = 'none'`.
    Similar to `Coupling` but applies only an affine transformation
    which here depends on time `t` such that it's identity map at `t = 0`.

    Args:
        latent_net (Type[nn.Module]): Inputs concatenation of `x` and `t` (and optionally
            `latent`) and outputs affine transformation parameters (size `2 * dim`)
        time_net (Type[stribor.net.time_net]): Time embedding with the same output
            size as `latent_net`
        mask (str): Mask name from `stribor.util.mask`
    """
    def __init__(
        self, 
        latent_net, 
        latent_net2, 
        time_net, 
        mask, 
        **kwargs
    ):
        super().__init__()

        self.latent_net = latent_net
        self.latent_net2 = latent_net2
        self.latent_net_h = kwargs.get('latent_net_h')
        self.mask_func = get_mask(mask) # Initializes mask generator
        self.time_net = time_net
        self.merge_scale = kwargs.get('merge_scale')
        self.merge_shift = kwargs.get('merge_shift')

    def get_mask(self, x):
        return self.mask_func(x.shape[-1]).expand_as(x).to(x)

    def forward(self, x, h, t, latent=None, reverse=False, **kwargs):
        """
        Args:
            x (tensor): Input with shape (..., dim)
            t (tensor): Time input with shape (..., 1)
            h: graph embedding
            latent (tensor): Conditioning vector with shape (..., latent_dim)
            reverse (bool, optional): Whether to calculate inverse. Default: False

        Returns:
            y (tensor): Transformed input with shape (..., dim)
            ljd (tensor): Log-Jacobian diagonal with shape (..., dim)
        """

        # data (x)
        mask = self.get_mask(x)
        z = torch.cat([x * 0 if x.shape[-1] == 1 else x * mask, t], -1)
        if latent is not None:
            z = torch.cat([z, latent], -1)

        scale, shift = self.latent_net(z).chunk(2, dim=-1)
        t_scale, t_shift = self.time_net(t).chunk(2, dim=-1)

        # graph augmented data (h)
        if h is not None:
            mask2 = self.get_mask(h)
            z2 = torch.cat([h * 0 if h.shape[-1] == 1 else h * mask2, t], -1)
            if latent is not None:
                z2 = torch.cat([z2, latent], -1)
            print(f"{self.latent_net_h=}")
            scale2, shift2 = self.latent_net_h(z2).chunk(2, dim=-1)
            merged_scale = self.merge_scale(torch.cat([scale, scale2], -1))
            merged_shift = self.merge_shift(torch.cat([shift, shift2], -1))
        else:
            merged_scale, merged_shift = scale, shift

        if reverse:
            y = (x - shift * t_shift) * torch.exp(-scale * t_scale)
        else:
            y = x * torch.exp(merged_scale * t_scale) + merged_shift * t_shift

        y = y * (1 - mask) + x * mask
        return y

