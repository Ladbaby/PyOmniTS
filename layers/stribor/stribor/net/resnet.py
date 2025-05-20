from typing import List

import layers.stribor.stribor as st
import torch.nn as nn


class ResNetBlock(nn.Module):
    """
    Single ResNet block `y = x + g(x)`.

    Args:
        dim (int): Input and output size
        hidden_dims (List[int]): Hidden dimensions
        activation (str, optional): Activation function from `torch.nn`.
            Default: 'ReLU'
        final_activation (str, optional): Last activation. Default: None
    """

    def __init__(self, dim, hidden_dims, activation='ReLU',
        final_activation=None, **kwargs):
        super().__init__()
        self.net = st.net.MLP(dim, hidden_dims, dim, activation,
            final_activation)

    def forward(self, x):
        return x + self.net(x)


class ResNet(nn.Module):
    """
    ResNet - neural network consisting of residual layers.
    "Deep Residual Learning for Image Recognition" (https://arxiv.org/abs/1512.03385)

    Args:
        dim (int): Input and output size
        hidden_dims (List[int]): Hidden dimensions
        num_layers (int): Number of layers
        activation (str, optional): Activation function from `torch.nn`.
            Default: 'ReLU'
        final_activation (str, optional): Last activation. Default: None
    """

    def __init__(self, dim, hidden_dims, num_layers, activation='ReLU',
        final_activation=None, **kwargs):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(ResNetBlock(dim, hidden_dims, activation,
                final_activation))
        self.net = nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        return self.net(x)
