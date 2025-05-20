from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module

from layers.GNeuralFlow.models.gnn import GNN
from layers.GNeuralFlow.models.h_coupling_flow import ContinuousAffineCoupling
from layers.GNeuralFlow.models.mlp import MLP, MlpLN
from layers.GNeuralFlow.models.resnet_flow import ResNetFlowNF
from layers.GNeuralFlow.train_utils import set_seed
import layers.GNeuralFlow.models as mods

class Flow(nn.Module):
    """
    Building normalizing flows, neural flows, and GNeuralFlow.
    """
    def __init__(self, base_dist=None, transforms=[]):
        super().__init__()
        self.base_dist = base_dist
        self.transforms = nn.ModuleList(transforms)

    def forward(self, x, h=None, latent=None, mask=None, t=None, reverse=False, **kwargs):
        """
        Args:
            x (tensor): Input sampled from base density with shape (..., dim)
            latent (tensor, optional): Conditional vector with shape (..., latent_dim)
                Default: None
            h: graph embedding
            mask (tensor): Masking tensor with shape (..., 1)
                Default: None
            t (tensor, optional): Flow time end point. Default: None
            reverse (bool, optional): Whether to perform an inverse. Default: False

        Returns:
            y (tensor): Output that follows target density (..., dim)
            log_jac_diag (tensor): Log-Jacobian diagonal (..., dim)
        """
        transforms = self.transforms[::-1] if reverse else self.transforms
        _mask = 1 if mask is None else mask

        for f in transforms:
            x = f.forward(x * _mask, h, latent=latent, mask=mask, t=t, **kwargs)
        return x

    def log_prob(self, x, **kwargs):
        """
        Calculates log-probability of a sample.

        Args:
            x (tensor): Input with shape (..., dim)

        Returns:
            log_prob (tensor): Log-probability of the input with shape (..., 1)
        """
        if self.base_dist is None:
            raise ValueError('Please define `base_dist` if you need log-probability')
        x, log_jac_diag = self.inverse(x, **kwargs)
        log_prob = self.base_dist.log_prob(x) + log_jac_diag.sum(-1)
        return log_prob.unsqueeze(-1)

    def sample(self, num_samples, latent=None, mask=None, **kwargs):
        """
        Transforms samples from the base to the target distribution.
        Uses reparametrization trick.

        Args:
            num_samples (tuple or int): Shape of samples
            latent (tensor): Latent conditioning vector with shape (..., latent_dim)

        Returns:
            x (tensor): Samples from target distribution with shape (*num_samples, dim)
        """
        if self.base_dist is None:
            raise ValueError('Please define `base_dist` if you need sampling')
        if isinstance(num_samples, int):
            num_samples = (num_samples,)

        x = self.base_dist.rsample(num_samples)
        x, log_jac_diag = self.forward(x, **kwargs)
        return x


class CouplingFlow_latent(Module):
    """
    Affine coupling flow

    Args:
        dim: Data dimension
        n_layers: Number of flow layers
        hidden_dims: Hidden dimensions of the flow neural network
        time_net: Time embedding module
        time_hidden_dim: Time embedding hidden dimension
    """
    def __init__(
        self,
        dim: int,
        n_layers: int,
        hidden_dims: List[int],
        time_net: Module,
        time_hidden_dim: Optional[int] = None,
        **kwargs
    ):
        super().__init__()

        transforms = []
        for i in range(n_layers):
            transforms.append(ContinuousAffineCoupling(
                latent_net=MlpLN(dim + 1, hidden_dims, 2 * dim, **kwargs),
                latent_net2=MlpLN((dim * 2), hidden_dims, dim, **kwargs),
                latent_net_h=MlpLN((dim + 1), hidden_dims, 2 * dim, **kwargs),
                merge_scale=MLP(dim * 2, hidden_dims, dim),
                merge_shift=MLP(dim * 2, hidden_dims, dim),
                time_net=getattr(mods, time_net)(2 * dim, hidden_dim=time_hidden_dim),
                mask='none' if dim == 1 else f'ordered_{i % 2}',
                **kwargs
            ))

        self.flow = Flow(transforms=transforms)

    def forward(
        self,
        x: Tensor, # Initial conditions, (..., 1, dim)
        h: Tensor,
        t: Tensor, # Times to solve at, (..., seq_len, dim)
        t0: Optional[Tensor] = None,
    ) -> Tensor: # Solutions to IVP given x at t, (..., times, dim)

        if x.shape[-2] == 1:
            x = x.repeat_interleave(t.shape[-2], dim=-2)
        if h is not None and h.shape[-2] == 1:
            h = h.repeat_interleave(t.shape[-2], dim=-2)

        # If t0 not 0, solve inverse first
        if t0 is not None:
            x = self.flow.inverse(x, t=t0)[0]

        return self.flow(x, h, t=t)


class CouplingFlow(Module):
    """
    Affine coupling flow

    Args:
        dim: Data dimension
        n_layers: Number of flow layers
        hidden_dims: Hidden dimensions of the flow neural network
        time_net: Time embedding module
        time_hidden_dim: Time embedding hidden dimension
    """
    def __init__(
        self,
        dim: int,
        n_layers: int,
        hidden_dims: List[int],
        time_net: Module,
        time_hidden_dim: Optional[int] = None,
        **kwargs
    ):
        super().__init__()

        transforms = []
        for i in range(n_layers):
            # transforms.append(st.ContinuousAffineCoupling(
            transforms.append(ContinuousAffineCoupling(
                latent_net=MLP(dim + 1, hidden_dims, 2 * dim),
                latent_net2=MLP((dim * 2), hidden_dims, dim),
                time_net=getattr(mods, time_net)(2 * dim, hidden_dim=time_hidden_dim),
                mask='none' if dim == 1 else f'ordered_{i % 2}'))

        self.flow = Flow(transforms=transforms)

    def forward(
        self,
        x: Tensor, # Initial conditions, (..., 1, dim)
        h: Tensor,
        t: Tensor, # Times to solve at, (..., seq_len, dim)
        t0: Optional[Tensor] = None,
    ) -> Tensor: # Solutions to IVP given x at t, (..., times, dim)

        if x.shape[-2] == 1:
            x = x.repeat_interleave(t.shape[-2], dim=-2)
        if h is not None and h.shape[-2] == 1:
            h = h.repeat_interleave(t.shape[-2], dim=-2)

        # If t0 not 0, solve inverse first
        if t0 is not None:
            x = self.flow.inverse(x, t=t0)[0]

        return self.flow(x, h, t=t)


class ResNetFlow(Module):
    """
    ResNet flow

    Args:
        dim: Data dimension
        n_layers: Number of flow layers
        hidden_dims: Hidden dimensions of the residual neural network
        time_net: Time embedding module
        time_hidden_dim: Time embedding hidden dimension
        invertible: Whether to make ResNet invertible (necessary for proper flow)
    """
    def __init__(
        self,
        dim: int,
        n_layers: int,
        hidden_dims: List[int],
        time_net: str,
        time_hidden_dim: Optional[int] = None,
        invertible: Optional[bool] = True,
        gnn_layers: int = 1,
        **kwargs
    ):
        super().__init__()

        layers = []
        for _ in range(n_layers):
            layers.append(ResNetFlowNF(
            # layers.append(st.net.ResNetFlow(
                dim,
                hidden_dims,
                n_layers,
                activation='ReLU',
                final_activation=None,
                time_net=time_net,
                time_hidden_dim=time_hidden_dim,
                invertible=invertible
            ))

        self.layers = nn.ModuleList(layers)
        self.data = kwargs.get('data')
        features = {'sink': 2, 'ellipse': 2, 'activity': 3}
        nfeats = features[self.data] if self.data in list(features.keys()) else 1
        self.nfeats = nfeats

        gcn_layers = []
        for _ in range(gnn_layers):
            gcn_layers.append(GNN(input_size=self.nfeats, hidden_size=32))
        self.gcn = torch.nn.ModuleList(gcn_layers)

        self.emb = MLP(in_dim=self.nfeats,
                       hidden_dims=hidden_dims,
                       out_dim=self.nfeats,
                       activation='ReLU',
                       final_activation='Tanh',
                       wrapper_func=None)

    def get_weights(self, _net):
        return [v for k, v in _net.named_parameters() if 'weight' in k]

    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0.
        # for net in [self.emb, self.layers, self.gcn]:
        for net in [self.gcn]:
            w = self.get_weights(net)
            for _w in w:
                reg += torch.sum(_w ** 2)
        return reg

    def l1_reg(self):
        """Take l1 norm of all parameters"""
        reg = 0.
        # for net in [self.emb, self.layers, self.gcn]:
        for net in [self.gcn]:
            w = self.get_weights(net)
            for _w in w:
                reg += torch.sum(_w)
        return reg

    def forward(self, x, h, t):
        if h is not None:
            if h.shape[-2] == 1:
                h = h.repeat_interleave(t.shape[-2], dim=-2)
        if x.shape[-2] == 1:
            x = x.repeat_interleave(t.shape[-2], dim=-2)
        for layer in self.layers:
            x = layer(x, h, t)

        return x
