from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module
from torchdiffeq import odeint_adjoint as odeint


class DiffeqConcat(Module):
    """
    Drift function for neural ODE model

    Args:
        dim: Data dimension
        hidden_dims: Hidden dimensions of the neural network
        activation: Name of the activation function from `torch.nn`
        final_activation: Name of the activation function from `torch.nn`
    """
    def __init__(
        self,
        dim: int,
        hidden_dims: List[int],
        activation: str,
        final_activation: str,
    ):
        super().__init__()
        import stribor as st
        self.net = st.net.MLP(dim + 1, hidden_dims, dim, activation, final_activation)

    def forward(
        self,
        t: Tensor, # Time point, scalar
        state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """ Input: t: (), state: tuple(x (..., n, d), diff (..., n, 1)) """
        x, diff = state
        x = torch.cat([t * diff, x], -1)
        dx = self.net(x) * diff
        return dx, torch.zeros_like(diff).to(dx)


class ODEModel(Module):
    """
    Neural ordinary differential equation model
    Implements reparameterization and seminorm trick for ODEs

    Args:
        dim: Data dimension
        net: Either a name (only `concat` supported) or a torch.Module
        hidden_dims: Hidden dimensions of the neural network
        activation: Name of the activation function from `torch.nn`
        final_activation: Name of the activation function from `torch.nn`
        solver: Which numerical solver to use (e.g. `dopri5`, `euler`, `rk4`)
        solver_step: How many solvers steps to take, only applicable for fixed step solvers
        atol: Absolute tolerance
        rtol: Relative tolerance
    """
    def __init__(
        self,
        dim: int,
        net: Union[str, Module],
        hidden_dims: List[int],
        activation: str,
        final_activation: str,
        solver: str,
        solver_step: Optional[int] = None,
        atol: Optional[float] = 1e-4,
        rtol: Optional[float] = 1e-3,
    ):
        super().__init__()

        self.atol = atol
        self.rtol = rtol

        if net == 'concat':
            self.net = DiffeqConcat(dim, hidden_dims, activation, final_activation)
        elif isinstance(net, Module):
            self.net = net
        else:
            raise NotImplementedError

        self.solver = solver

        if solver == 'dopri5':
            self.options = None
        else:
            self.options = { 'step_size': solver_step }

    def forward(
        self,
        x: Tensor, # Initial conditions, (..., 1, dim)
        t: Tensor, # Times to solve at, (..., seq_len, dim)
    ) -> Tensor: # Solutions to IVP given x at t, (..., seq_len, dim)

        if x.shape[-2] == 1:
            x = x.repeat_interleave(t.shape[-2], dim=-2)

        y = odeint(
            self.net, # Drift network
            (x, t), # Initial condition
            torch.Tensor([0, 1]).to(x), # Reparameterization trick
            method=self.solver,
            options=self.options,
            atol=self.atol,
            rtol=self.rtol,
            adjoint_options=dict(norm='seminorm') # Seminorm trick
        )[0][1] # get first state (x), second output (at t=1)

        return y
