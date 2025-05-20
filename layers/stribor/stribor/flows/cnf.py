from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint, odeint
import layers.stribor.stribor as st
from layers.stribor.stribor import Transform
__all__ = ['ContinuousTransform']


class ODEfunc(nn.Module):
    """
    ODE function used in continuous normalizing flows.
    Based on FFJORD (https://arxiv.org/abs/1810.01367)
    Code adapted from https://github.com/rtqichen/ffjord

    Args:
        diffeq (nn.Module): Given inputs `x` and `t`, returns `dx` (and optionally trace)
        divergence (str): How to calculate divergence.
            Options: 'compute', 'compute_set', 'approximate', 'exact', 'none'
        rademacher (bool, optional): Whether to use Rademacher distribution for stochastic
            estimator, otherwise uses normal distribution. Default: False
        has_latent (bool, optional): Whether we have latent inputs. Default: False
        set_data (bool, optional): Whether we have set data with shape (..., N, dim). Default: False
    """

    def __init__(self, diffeq, divergence=None, rademacher=False,
        has_latent=False, set_data=False, **kwargs):
        super().__init__()
        assert divergence in ['compute', 'compute_set', 'approximate',
            'exact', 'none']
        self.diffeq = diffeq
        self.rademacher = rademacher
        self.divergence = divergence
        self.has_latent = has_latent
        self.set_data = set_data
        self.register_buffer('_num_evals', torch.tensor(0.0))

    def before_odeint(self, e=None):
        self._e = e
        self._num_evals.fill_(0)

    def num_evals(self):
        return self._num_evals.item()

    def forward(self, t, states):
        self._num_evals += 1
        y = states[0]
        t = torch.Tensor([t]).to(y)
        latent = mask = None
        if len(states) == 4:
            latent = states[2]
            mask = states[3]
        elif len(states) == 3:
            if self.has_latent:
                latent = states[2]
            else:
                mask = states[2]
        if self._e is None and self.divergence == 'approximate':
            if self.rademacher:
                self._e = torch.randint(low=0, high=2, size=y.shape).to(y
                    ) * 2 - 1
            else:
                self._e = torch.randn_like(y)
        if self.divergence == 'none':
            dy = self.diffeq(t, y, latent=latent, mask=mask)
            return (dy,) + tuple(torch.zeros_like(s) for s in states[2:])
        if self.divergence == 'exact':
            dy, divergence = self.diffeq(t, y, latent=latent, mask=mask)
        else:
            with torch.set_grad_enabled(True):
                y.requires_grad_(True)
                dy = self.diffeq(t, y, latent=latent, mask=mask)
                if not self.training or 'compute' in self.divergence:
                    if self.set_data or self.divergence == 'compute_set':
                        divergence = st.util.divergence_exact_for_sets(dy, y)
                    else:
                        divergence = st.util.divergence_exact(dy, y)
                else:
                    divergence = st.util.divergence_approx(dy, y, self._e)
        return (dy, -divergence) + tuple(torch.zeros_like(s) for s in
            states[2:])


class ContinuousTransform(Transform):
    """
    Continuous normalizing flow transformation.
    "Neural Ordinary Differential Equations" (https://arxiv.org/abs/1806.07366)

    Example:
    >>> torch.manual_seed(123)
    >>> dim = 2
    >>> f = stribor.ContinuousTransform(dim, net=stribor.net.DiffeqMLP(dim + 1, [64], dim))
    >>> f(torch.randn(1, dim))
    (tensor([[-0.1527, -0.4164]], tensor([[-0.1218, -0.7133]])

    Args:
        dim (int): Input data dimension
        net (nn.Module): Neural net that defines a differential equation.
            It takes `x` and `t` and returns `dx` (and optionally trace).
        T (float): Upper bound of integration. Default: 1.0
        divergence: How to calculate divergence. Options:
            `compute`: Brute force approach, scales with O(d^2)
            `compute_set`: Same as compute but for densities over sets with shape (..., N, dim)
            `approximate`: Stochastic estimator, only used during training, O(d)
            `exact`: Exact trace, returned from the `net`
            Default: 'approximate'
        use_adjoint (bool, optional): Whether to use adjoint method for backpropagation,
            which is more memory efficient. Default: True
        solver (string): ODE black-box solver.
            adaptive: `dopri5`, `dopri8`, `bosh3`, `adaptive_heun`
            exact-step: `euler`, `midpoint`, `rk4`, `explicit_adams`, `implicit_adams`
            Default: 'dopri5'
        solver_options (dict, optional): Additional options, e.g. `{'step_size': 10}`. Default: {}
        test_solver (str, optional): Which solver to use during evaluation. If not specified,
            uses the same as during training. Default: None
        test_solver_options (dict, optional): Which solver options to use during evaluation.
            If not specified, uses the same as during training. Default: None
        set_data (bool, optional): If data is of shape (..., N, D). Default: False
        rademacher (bool, optional): Whether to use Rademacher distribution for stochastic
            estimator, otherwise uses normal distribution. Default: False
        atol (float): Absolute tolerance (Default: 1e-5)
        rtol (float): Relative tolerance (Default: 1e-3)
    """

    def __init__(self, dim, net=None, T=1.0, divergence='approximate',
        use_adjoint=True, has_latent=False, solver='dopri5', solver_options
        ={}, test_solver=None, test_solver_options=None, set_data=False,
        rademacher=False, atol=1e-05, rtol=0.001, **kwargs):
        super().__init__()
        self.T = T
        self.dim = dim
        self.odefunc = ODEfunc(net, divergence, rademacher, has_latent,
            set_data)
        self.integrate = odeint_adjoint if use_adjoint else odeint
        self.solver = solver
        self.solver_options = solver_options
        self.test_solver = test_solver or solver
        self.test_solver_options = (solver_options if test_solver_options is
            None else test_solver_options)
        self.atol = atol
        self.rtol = rtol

    def forward_and_log_det_jacobian(self, x, latent=None, mask=None, *,
        reverse: Optional[bool]=False, **kwargs):
        log_trace_jacobian = torch.zeros_like(x)
        integration_times = torch.tensor([0.0, self.T]).to(x)
        if reverse:
            integration_times = _flip(integration_times, 0)
        self.odefunc.before_odeint()
        initial = x, log_trace_jacobian
        if latent is not None:
            initial += latent,
        if mask is not None:
            initial += mask,
        state_t = self.integrate(self.odefunc, initial, integration_times,
            atol=self.atol, rtol=self.rtol, method=self.solver if self.
            training else self.test_solver, options=self.solver_options if
            self.training else self.test_solver_options)
        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)
        x, log_trace_jacobian = state_t[:2]
        return x, -log_trace_jacobian.sum(-1, keepdim=True)

    def inverse_and_log_det_jacobian(self, y, latent=None, mask=None, **kwargs
        ):
        return self.forward_and_log_det_jacobian(y, latent, mask, reverse=True)

    def forward(self, x, latent=None, mask=None, **kwargs):
        y, _ = self.forward_and_log_det_jacobian(x, latent, mask)
        return y

    def inverse(self, y, latent=None, mask=None, **kwargs):
        x, _ = self.inverse_and_log_det_jacobian(y, latent, mask)
        return x

    def log_det_jacobian(self, x, y, mask=None, latent=None, **kwargs):
        _, log_det_jacobian = self.forward_and_log_det_jacobian(x, latent, mask
            )
        return log_det_jacobian

    def _num_evals(self):
        return self.odefunc._num_evals.item()


def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long,
        device=x.device)
    return x[tuple(indices)]
