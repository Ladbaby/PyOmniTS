from typing import List, Optional

import torch.nn as nn


class EquivariantLayer(nn.Module):
    """
    Permutation equivariant layer. Computes the transformation of each
    set element conditioned on the sum of all elements.
    Similar to "Deep Sets" (https://arxiv.org/abs/1703.06114) and
    "On Learning Sets of Symmetric Elements" (https://arxiv.org/abs/2002.08599).

    Args:
        in_dim (int): Dimension of input
        out_dim (int): Dimension of output
    """

    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()
        self.l1 = nn.Linear(in_dim, out_dim)
        self.l2 = nn.Linear(in_dim, out_dim)

    def forward(self, x, mask=None, **kwargs):
        y1 = self.l1(x)
        y2 = self.l2(x.sum(-2, keepdim=True))
        if mask is not None:
            mask = mask[..., 0, None]
            y1 = y1 * mask
            y2 = y2 * mask / mask.sum(-2, keepdim=True)
        else:
            y2 = y2 / x.shape[-2]
        return y1 + y2


class EquivariantNet(nn.Module):
    """
    Neural network with permutation equivariant layers.
    Takes sets of elements of shape (..., N, dim). Permuting the elements across
    the second to last dimension results in the same permutation on the output.
    Args similar to `st.net.MLP` but uses `EquivariantLayer` instead of `nn.Linear`.

    Example:
    >>> net = stribor.net.EquivariantNet(2, [64, 64], 4)

    Args:
        in_dim (int): Input size
        hidden_dims (List[int]): Hidden dimensions
        out_dim (int): Output size
        activation (str, optional): Activation function from `torch.nn`. Default: 'Tanh'
        final_activation (str, optional): Last activation. Default: None
    """

    def __init__(self, in_dim, hidden_dims, out_dim, activation='Tanh',
        final_activation=None, **kwargs):
        super().__init__()
        self.activation = getattr(nn, activation)()
        self.final_activation = getattr(nn, final_activation)(
            ) if final_activation else nn.Identity()
        hidden_dims = [in_dim] + hidden_dims + [out_dim]
        self.layers = []
        for in_, out_ in zip(hidden_dims[:-1], hidden_dims[1:]):
            self.layers.append(EquivariantLayer(in_, out_))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x, mask, **kwargs):
        for layer in self.layers[:-1]:
            x = layer(x, mask=mask)
            x = self.activation(x)
        x = self.layers[-1](x, mask=mask)
        x = self.final_activation(x)
        return x
