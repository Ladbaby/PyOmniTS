from typing import List, Union, Optional, Tuple

import torch.nn as nn
import layers.stribor.stribor as st
from layers.stribor.stribor.net import DiffeqNet


class DiffeqExactTrace(nn.Module):
    """
    Differential equation with exact trace.
    Recommended to directly use tested implementations, e.g., `DiffeqExactTraceMLP`.

    Args:
        exclusive_net (st.net.DiffeqNet): Hollow Jacobian net, e.g., `st.net.DiffeqZeroTraceMLP`
        dimwise_net (st.net.DiffeqNet): Per dimension net, e.g,. `st.net.DiffeqMLP`
    """

    def __init__(self, exclusive_net, dimwise_net, return_log_det_jac=True,
        **kwargs):
        super().__init__()
        self.exclusive_net = exclusive_net
        self.dimwise_net = dimwise_net
        self.return_log_det_jac = return_log_det_jac

    def forward(self, t, x, latent=None, **kwargs):
        params = st.util.flatten_params(self.exclusive_net, self.dimwise_net)
        y, jac = st.net.FuncAndDiagJac.apply(self.exclusive_net, self.
            dimwise_net, t, x, latent, params)
        return (y, jac) if self.return_log_det_jac else y


class DiffeqExactTraceMLP(DiffeqExactTrace):
    """
    Exact trace network that mimics MLP architecture.

    Args:
        in_dim (int): Input size
        hidden_dims (List[int]): Hidden dimensions
        out_dim (int): Output size
        d_h (int): Size of conditioning vector per each dimension
        latent_dim (int, optional): Size of latent vector. Default: 0
        return_log_det_jac (bool, optional): Whether to return log-Jacobian diagonal. Default: True
    """

    def __init__(self, in_dim, hidden_dims, out_dim, d_h, latent_dim=0,
        return_log_det_jac=True, **kwargs):
        exclusive_net = st.net.DiffeqZeroTraceMLP(in_dim, hidden_dims, d_h *
            out_dim, return_log_det_jac=False, return_per_dim=True)
        dimwise_net = st.net.DiffeqMLP(d_h + latent_dim + 2, hidden_dims, 1)
        super().__init__(exclusive_net, dimwise_net, return_log_det_jac)


class DiffeqExactTraceDeepSet(DiffeqExactTrace):
    """
    Exact trace network that mimics deepset architecture.

    Args:
        in_dim (int): Input size
        hidden_dims (List[int]): Hidden dimensions
        out_dim (int): Output size
        d_h (int): Size of conditioning vector per each dimension
        latent_dim (int, optional): Size of latent vector. Default: 0
        pooling (str): Pooling operation. Default: 'max'
        return_log_det_jac (bool, optional): Whether to return log-Jacobian diagonal. Default: True
    """

    def __init__(self, in_dim, hidden_dims, out_dim, d_h, latent_dim=0,
        pooling='max', return_log_det_jac=True, **kwargs):
        exclusive_net = st.net.DiffeqZeroTraceDeepSet(in_dim, hidden_dims, 
            d_h * out_dim, return_log_det_jac=False)
        dimwise_net = st.net.DiffeqMLP(d_h + latent_dim + 2, hidden_dims, 1)
        super().__init__(exclusive_net, dimwise_net, return_log_det_jac)


class DiffeqExactTraceAttention(DiffeqExactTrace):
    """
    Exact trace network that mimics deepset architecture.

    Args:
        in_dim (int): Input size
        hidden_dims (List[int]): Hidden dimensions
        out_dim (int): Output size
        d_h (int): Size of conditioning vector per each dimension
        latent_dim (int, optional): Size of latent vector. Default: 0
        n_heads (int, optional): Number of attention heads. Default: 1
        return_log_det_jac (bool, optional): Whether to return log-Jacobian diagonal. Default: True
    """

    def __init__(self, in_dim, hidden_dims, out_dim, d_h, latent_dim=0,
        n_heads=1, return_log_det_jac=True, **kwargs):
        exclusive_net = st.net.DiffeqZeroTraceAttention(in_dim, hidden_dims,
            d_h * out_dim, n_heads, return_log_det_jac=False)
        dimwise_net = st.net.DiffeqMLP(d_h + latent_dim + 2, hidden_dims, 1)
        super().__init__(exclusive_net, dimwise_net, return_log_det_jac)
