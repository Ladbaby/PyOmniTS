from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module

from layers.GNeuralFlow.models import ODEModel
import layers.GNeuralFlow.models as mods
from layers.GNeuralFlow.models.mlp import *
import layers.GNeuralFlow.models.mlp as mlps


class GRUODENet(Module):
    """
    GRU-ODE drift function

    Args:
        hidden_dim: Size of the GRU hidden state
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.lin_hh = nn.Linear(hidden_dim, hidden_dim)
        self.lin_hz = nn.Linear(hidden_dim, hidden_dim)
        self.lin_hr = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        t: Tensor,
        inp: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:

        h, diff = inp[0], inp[1]

        # Continuous gate functions
        r = torch.sigmoid(self.lin_hr(h))
        z = torch.sigmoid(self.lin_hz(h))
        u = torch.tanh(self.lin_hh(r * h))

        # Final drift
        dh = (1 - z) * (u - h) * diff

        return dh, torch.zeros_like(diff).to(dh)


class GRUFlowBlock_LN(Module):
    """
    Single GRU flow layer

    Args:
        hidden_dim: Size of the GRU hidden state
        time_net: Time embedding module
        time_hidden_dim: Time embedding hidden dimension
    """
    def __init__(
        self,
        hidden_dim,
        time_net,
        time_hidden_dim=None
    ):
        super().__init__()

        # Spectral norm for linear layers
        norm = lambda layer: torch.nn.utils.spectral_norm(layer, n_power_iterations=5)

        self.lin_hh = norm(nn.Linear(hidden_dim + 1, hidden_dim))
        self.lin_hz = norm(nn.Linear(hidden_dim + 1, hidden_dim))
        self.lin_hr = norm(nn.Linear(hidden_dim + 1, hidden_dim))
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)

        self.time_net = getattr(mods, time_net)(hidden_dim, hidden_dim=time_hidden_dim)

        # Additional constants that ensure invertibility, see Theorem 1 in Bilos et al. 2021
        self.alpha = 2 / 5
        self.beta = 4 / 5

    def residual(self, h, t):
        inp = torch.cat([h, t], -1)
        r = self.beta * torch.sigmoid(self.ln1(self.lin_hr(inp)))
        z = self.alpha * torch.sigmoid(self.ln2(self.lin_hz(inp)))
        u = torch.tanh(self.ln3(self.lin_hh(torch.cat([r * h, t], -1))))
        return z * (u - h)

    def forward(self, x, h, t):
        res_x = self.residual(x, t)
        x = x + self.time_net(t) * res_x
        return x

    def inverse(self, y, t, iterations=100):
        x = y
        for _ in range(iterations):
            residual = self.time_net(t) * self.residual(x, t)
            x = y - residual
        return x


class GRUFlowBlock(Module):
    """
    Single GRU flow layer

    Args:
        hidden_dim: Size of the GRU hidden state
        time_net: Time embedding module
        time_hidden_dim: Time embedding hidden dimension
    """
    def __init__(
        self,
        hidden_dim,
        time_net,
        time_hidden_dim=None,
        **kwargs
    ):
        super().__init__()

        # Spectral norm for linear layers
        norm = lambda layer: torch.nn.utils.spectral_norm(layer, n_power_iterations=5)

        self.lin_hh2 = norm(nn.Linear(hidden_dim + 1, hidden_dim))
        self.lin_hz2 = norm(nn.Linear(hidden_dim + 1, hidden_dim))
        self.lin_hr2 = norm(nn.Linear(hidden_dim + 1, hidden_dim))

        self.time_net = getattr(mods, time_net)(hidden_dim, hidden_dim=time_hidden_dim)

        # Additional constants that ensure invertibility, see Theorem 1 in Bilos et al. 2021
        self.alpha = 2 / 5
        self.beta = 4 / 5

    def residual(self, h, t):
        inp = torch.cat([h, t], -1)
        r = self.beta * torch.sigmoid(self.lin_hr(inp))
        z = self.alpha * torch.sigmoid(self.lin_hz(inp))
        u = torch.tanh(self.lin_hh(torch.cat([r * h, t], -1)))
        return z * (u - h)

    def forward_orig(self, h, t):
        h = h + self.time_net(t) * self.residual(h, t)
        return h

    def residual2(self, h, x, t):
        inp = torch.cat([h, t], -1)
        r = self.beta * torch.sigmoid(self.lin_hr2(inp))
        z = self.alpha * torch.sigmoid(self.lin_hz2(inp))
        u = torch.tanh(self.lin_hh2(torch.cat([r * h, t], -1)))
        return z * (u - x)

    def forward(self, x, h, t):
        if h is not None:
            res_h = self.residual2(h, x, t)
            x = x + self.time_net(t) * res_h
        else:
            res_x = self.residual(x, t)
            x = x + self.time_net(t) * res_x
        return x

    def inverse(self, y, t, iterations=100):
        x = y
        for _ in range(iterations):
            residual = self.time_net(t) * self.residual(x, t)
            x = y - residual
        return x


class GRUFlow(Module):
    """
    GRU flow model

    Args:
        dim: Data dimension
        n_layers: Number of flow layers
        time_net: Time embedding module
        time_hidden_dim: Time embedding hidden dimension
    """
    def __init__(
        self,
        dim: int,
        n_layers: int,
        time_net: str,
        time_hidden_dim: Optional[int] = None,
        **kwargs
    ):
        super().__init__()

        layers = []
        for _ in range(n_layers):
            layers.append(GRUFlowBlock(dim, time_net, time_hidden_dim, **kwargs))

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x: Tensor, h: Tensor, t: Tensor) -> Tensor:
        if x.shape[-2] != t.shape[-2]:
            x = x.repeat_interleave(t.shape[-2], dim=-2)
        if h is not None and h.shape[-2] != t.shape[-2]:
            h = h.repeat_interleave(t.shape[-2], dim=-2)

        for layer in self.layers:
            x = layer(x, h, t)

        return x

    def inverse(self, y, t):
        for layer in reversed(self.layers):
            y = layer.inverse(y, t)
        return y


class ContinuousGRULayer(Module):
    """
    Continuous GRU layer

    Args:
        dim: Data dimension
        hidden_dim: GRU hidden dimension
        model: Which model to use (`ode` or `flow`)
        flow_model: Which flow model to use (currently only `resnet` supported which gives GRU flow)
        flow_layers: How many flow layers
        time_net: Time embedding module
        time_hidden_dim: Time embedding hidden dimension
        solver: Which numerical solver to use
        solver_step: How many solvers steps to take, only applicable for fixed step solvers
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        model: str,
        flow_model: Optional[str] = None,
        flow_layers: Optional[int] = None,
        time_net: Optional[str] = None,
        time_hidden_dim: Optional[int] = None,
        solver: Optional[str] = None,
        solver_step: Optional[int] = None,
        **kwargs
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        if model == 'ode':
            self.odeint = ODEModel(hidden_dim, GRUODENet(hidden_dim), None, None, None, solver, solver_step)
        elif model == 'flow' and flow_model == 'resnet':
            self.odeint = GRUFlow(hidden_dim, flow_layers, time_net, time_hidden_dim)
        else:
            raise NotImplementedError

        self.gru = nn.GRU(dim, hidden_dim, 1, batch_first=True)

    def forward(
        self,
        x: Tensor, # Initial conditions, (..., seq_len, dim)
        t: Tensor, # Times to solve at, (..., seq_len, dim)
    ) -> Tensor: # Solutions to IVP given x at t, (..., seq_len, dim)

        # Initial hidden state
        h = torch.zeros(1, 1, self.hidden_dim).repeat(x.shape[0], 1, 1).to(x)

        # Hidden (pre-jump) states will be stored here
        hiddens = torch.zeros(*x.shape[:-1], self.hidden_dim).to(x)

        for i in range(t.shape[1]):
            # Evolve the hidden state

            h = self.odeint(h, t[:,i,None])
            hiddens[:,i,None] = h

            # Update the hidden state with observation
            _, h = self.gru(x[:,i,None], h.transpose(0, 1))
            h = h.transpose(0, 1)

        return hiddens

