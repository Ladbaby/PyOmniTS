# import stribor as st
import torch
import torch.nn as nn

from layers.GNeuralFlow.models.gnn import GNN
from layers.GNeuralFlow.models.mlp import MLP
import layers.GNeuralFlow.models as mods


class ResNetFlowBlock(nn.Module):
    def __init__(self, dim, hidden_dims, activation, final_activation, time_net,
                 time_hidden_dim, n_power_iterations, invertible=True, **kwargs):
        super().__init__()
        self.invertible = invertible
        wrapper = None

        if invertible:
            wrapper = lambda layer: torch.nn.utils.spectral_norm(layer, n_power_iterations=n_power_iterations)

        self.net = MLP(dim * 2 + 1, hidden_dims, dim, activation, final_activation, wrapper_func=wrapper)
        self.net2 = MLP(dim + 1, [32, 32, 64, 32], dim, activation, final_activation, wrapper_func=wrapper)
        self.net3 = MLP(dim * 2, hidden_dims, dim, activation, final_activation, wrapper_func=wrapper)
        self.xh_net = MLP(dim*2, hidden_dims, dim, activation, final_activation, wrapper_func=wrapper)
        self.time_net = getattr(mods, time_net)(dim, hidden_dim=time_hidden_dim)

    def forward(self, x, h, t):
        t_output = self.time_net(t)
        if h is not None:
            x_output = x + t_output * (torch.mul(self.net(torch.cat([x, h, t], -1)), self.net2(torch.cat([x, t], -1))))
        else:
            x_output = x + t_output * self.net2(torch.cat([x, t], -1))
        return x_output

    def inverse(self, y, t, iterations=100):
        if not self.invertible:
            raise NotImplementedError
        # fixed-point iteration
        x = y
        for _ in range(iterations):
            residual = self.time_net(t) * self.net(torch.cat([x, t], -1))
            x = y - residual
        return x


class ResNetFlowNF(nn.Module):
    """
    ResNet flow. For a given input and time t, it returns a solution to some ODE.

    Example:
    >>> dim = 3
    >>> model = ResNetFlowNF(dim, [64], 1, time_net='TimeTanh')
    >>> x = torch.randn(32, dim)
    >>> t = torch.rand(32, 1)
    >>> model(x, t).shape
    torch.Size([32, 3])

    Args:
        dim (int): Input and output size
        hidden_dims (List[int]): Hidden dimensions
        num_layers (int): Number of layers
        activation (str, optional): Activation function from `torch.nn`.
            Default: 'ReLU'
        final_activation (str, optional): Last activation. Default: None
        time_net (str): Time embedding network from `stribor.net.time_net`
        time_hidden_dim (int, optional): Time embedding size
        n_power_iterations (float, optional): Number of power iterations. Default: 5
        invertible (bool, optional): Whether to have invertible transformation.
            Default: True
    """
    def __init__(
        self,
        dim,
        hidden_dims,
        num_layers,
        activation='ReLU',
        final_activation=None,
        time_net=None,
        time_hidden_dim=None,
        n_power_iterations=5,
        invertible=True,
        **kwargs
    ):
        super().__init__()
        blocks = []
        for _ in range(num_layers):
            blocks.append(ResNetFlowBlock(dim, hidden_dims, activation, final_activation, time_net,
                                          time_hidden_dim, n_power_iterations, invertible))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, h, t):
        for block in self.blocks:
            x = block(x, h, t)
        return x

    def inverse(self, y, t):
        for block in reversed(self.blocks):
            y = block.inverse(y, t)
        return y
