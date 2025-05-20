from typing import Union, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
__all__ = ['MADE']


class MaskedLinear(nn.Linear):
    """
    Same as torch.nn.Linear except has a mask on the weights.

    Args:
        in_features (int): Input data dimension
        out_features (int): Output data dimension
        bias (bool): Whether to use bias. Default: True
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    """
    Autoregressive transformation with a neural network, implemented using weight masking.
    "MADE: Masked Autoencoder for Distribution Estimation" (https://arxiv.org/abs/1502.03509)

    Args:
        in_dim (int): Input size
        hidden_dims (List[int]): Hidden dimensions
        out_dim (int): Output size. Has to be multiple of `in_dim` such that each dimension
            has assigned output dimensions. `return_per_dim` governs how the output looks like
        activation (str): Activation function from `torch.nn`. Default: 'Tanh'
        final_activation (str): Last activation. Default: None
        num_masks (int): Number of ordering ensembles. Default: 1
        natural_ordering (bool): Whether to use natural order, else random. Default: False
        reverse_ordering (bool): Whether to reverse the order. Default: False
        return_per_dim (bool): Whether to return in (..., in_dim, out_dim / in_dim) format,
            otherwise returns (..., out_dim). Default: False
    """

    def __init__(self, in_dim, hidden_dims, out_dim, activation='Tanh',
        final_activation=None, num_masks=1, natural_ordering=False,
        reverse_ordering=False, return_per_dim=False, **kwargs):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.final_activation = final_activation
        self.return_per_dim = return_per_dim
        self.natural_ordering = natural_ordering
        self.reverse_ordering = reverse_ordering
        self.num_masks = num_masks
        assert self.out_dim % self.in_dim == 0, 'out_dim must be integer multiple of in_dim'
        self.net = self._get_net()
        self.m = {}
        self.update_masks()

    def _get_net(self):
        net = []
        hs = [self.in_dim] + self.hidden_dims + [self.out_dim]
        for h0, h1 in zip(hs, hs[1:]):
            net.append(MaskedLinear(h0, h1))
            net.append(getattr(nn, self.activation)())
        net.pop()
        if self.final_activation is not None:
            net.append(getattr(nn, self.final_activation)())
        return nn.Sequential(*net)

    def update_masks(self):
        if self.m and self.num_masks == 1:
            return
        L = len(self.hidden_dims)
        rng = np.random.RandomState()
        if self.natural_ordering:
            self.m[-1] = np.arange(self.in_dim)
            if self.reverse_ordering:
                self.m[-1] = self.m[-1][::-1]
        else:
            self.m[-1] = rng.permutation(self.in_dim)
        for l in range(L):
            self.m[l] = rng.randint(self.m[l - 1].min(), max(self.in_dim - 
                1, 1), size=self.hidden_dims[l])
        masks = [(self.m[l - 1][:, None] <= self.m[l][None, :]) for l in
            range(L)]
        masks.append(self.m[L - 1][:, None] < self.m[-1][None, :])
        if self.out_dim > self.in_dim:
            k = int(self.out_dim / self.in_dim)
            masks[-1] = np.concatenate([masks[-1]] * k, axis=1)
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l, m in zip(layers, masks):
            l.set_mask(m)

    def forward(self, x, **kwargs):
        original_shape = x.shape
        x = x.view(-1, original_shape[-1])
        y = self.net(x)
        y = y.view(*original_shape[:-1], -1, self.in_dim).transpose(-1, -2)
        return y if self.return_per_dim else y.reshape(*y.shape[:-2], -1)
