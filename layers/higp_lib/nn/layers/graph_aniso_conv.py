from typing import Optional

import torch
from torch import nn, Tensor
from torch_geometric.nn import MessagePassing
from layers.tsl.nn.layers import Dense
from layers.tsl.nn.layers.ops import Activation


class GraphAnisoConv(MessagePassing):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 edge_dim: Optional[int] = None,
                 activation: str = 'leaky_relu'):
        super(GraphAnisoConv, self).__init__(aggr="add", node_dim=-2)

        self.in_channels = input_size
        self.out_channels = output_size

        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * input_size, output_size),
            Activation(activation),
            nn.Linear(output_size, output_size),
        )

        edge_dim = edge_dim or 0
        if edge_dim > 0:
            self.lin_edge = nn.Linear(edge_dim, output_size)
        else:
            self.register_parameter('lin_edge', None)

        self.gate_mlp = Dense(output_size, 1, activation='sigmoid')

        self.skip_conn = nn.Linear(input_size, output_size)
        self.activation = Activation(activation)

    def forward(self, x, edge_index, edge_attr: Optional[Tensor] = None):
        """"""
        if edge_attr is not None:
            if edge_attr.ndim == 1:  # accommodate for edge_index
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        out = self.activation(out + self.skip_conn(x))

        return out

    def message(self, x_i, x_j, edge_attr):
        mij = self.msg_mlp(torch.cat([x_i, x_j], -1))
        if edge_attr is not None:
            mij = mij + edge_attr
        return self.gate_mlp(mij) * mij

