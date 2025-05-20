from typing import List

from torch.nn.utils import spectral_norm
import torch
import layers.stribor.stribor as st
from layers.stribor.stribor import Transform


class IResNet(Transform):
    """
    Invertible ResNet transformation.

    Args:
        dim: Input dimension
        hidden_dims: Hidden dimensions
        activation: Activation between hidden layers
        final_activation: Final activation
        n_power_iterations: How many iterations to perform in `spectral_norm`
    """

    def __init__(self, dim, hidden_dims, activation='ReLU',
        final_activation=None, n_power_iterations=5, **kwargs):
        super().__init__()
        wrapper = lambda layer: spectral_norm(layer, n_power_iterations=
            n_power_iterations)
        self.net = st.net.MLP(dim, hidden_dims, dim, activation,
            final_activation, nn_linear_wrapper_func=wrapper)

    def forward(self, x):
        return x + self.net(x)

    def inverse(self, y, iterations=100, **kwargs):
        x = y
        for _ in range(iterations):
            residual = self.net(x)
            x = y - residual
        return x

    def log_det_jacobian(self, x, y, **kwargs):
        return NotImplementedError


class ContinuousIResNet(Transform):
    """
    Continuous time invertible ResNet transformation.

    Args:
        dim: Input dimension
        hidden_dims: Hidden dimensions
        activation: Activation between hidden layers
        final_activation: Final activation
        time_net: Time embedding network
        time_hidden_dim: Time embedding dimension
        n_power_iterations: How many iterations to perform in `spectral_norm`
    """

    def __init__(self, dim, hidden_dims, *, activation: str='ReLU',
        final_activation: str=None, time_net: torch.nn.Module=None,
        n_power_iterations: int=5, **kwargs):
        super().__init__()
        wrapper = lambda layer: spectral_norm(layer, n_power_iterations=
            n_power_iterations)
        self.net = st.net.MLP(dim, hidden_dims, dim, activation,
            final_activation, nn_linear_wrapper_func=wrapper)
        self.time_net = time_net

    def forward(self, x, t):
        return x + self.time_net(t) * self.net(x)

    def inverse(self, y, t, iterations=100, **kwargs):
        x = y
        for _ in range(iterations):
            residual = self.time_net(t) * self.net(x)
            x = y - residual
        return x

    def log_det_jacobian(self, x, y, **kwargs):
        return NotImplementedError
