import torch
from einops import rearrange
from torch import nn, Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.nn.pool import TopKPooling
from torch_geometric.typing import PairTensor
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_sparse import SparseTensor

from layers.tsl.nn.layers.graph_convs import GraphConv
from layers.tsl.utils import ensure_list


class GraphUNet(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 n_layers,
                 pooling_factor,
                 activation='relu'):
        super(GraphUNet, self).__init__()

        up_convs = []
        down_convs = []
        pool_layers = []
        down_convs.append(
            GraphConv(
                input_size=input_size,
                output_size=hidden_size,
                activation=activation,
                norm='sym'
            )
        )
        for i in range(n_layers):
            down_convs.append(
                GraphConv(
                    input_size=hidden_size,
                    output_size=hidden_size,
                    activation=activation,
                    norm='sym'
                )
            )
            up_convs.append(
                GraphConv(
                    input_size=2 * hidden_size,
                    output_size=hidden_size,
                    activation=activation,
                    norm='sym'
                )
            )
            pool_layers.append(
                TopKPooling(in_channels=hidden_size, ratio=pooling_factor)
            )

        self.up_convs = nn.ModuleList(up_convs)
        self.down_convs = nn.ModuleList(down_convs)
        self.pool_layers = nn.ModuleList(pool_layers)
        self.n_layers = n_layers

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        b, n, _ = x.size()
        x = rearrange(x, 'b n f -> (b n) f')
        disjoint_graph = Batch.from_data_list([Data(edge_index=edge_index,
                                                    edge_attr=edge_weight,
                                                    num_nodes=n), ] * b)
        edge_index, edge_weight = disjoint_graph.edge_index, disjoint_graph.edge_attr
        node_idx = torch.arange(n, device=x.device).repeat_interleave(b)
        x = self.down_convs[0](x, edge_index, edge_weight)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.n_layers + 1):
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight,
                                                       x.size(0))
            x, edge_index, edge_weight, node_idx, perm, _ = self.pool_layers[i - 1](
                x, edge_index, edge_weight, batch=node_idx)

            x = self.down_convs[i](x, edge_index, edge_weight)

            if i < self.n_layers:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]

        for i in range(self.n_layers):
            j = self.n_layers - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            up = torch.zeros_like(res)
            up[perm] = x
            x = torch.cat([res, up], -1)

            x = self.up_convs[i](x, edge_index, edge_weight)

        x = rearrange(x, '(b n) f -> b n f', b=b, n=n)
        return x

    def augment_adj(self, edge_index: Tensor, edge_weight: Tensor,
                    num_nodes: int) -> PairTensor:
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        adj = SparseTensor.from_edge_index(edge_index, edge_weight,
                                           sparse_sizes=(num_nodes, num_nodes))
        adj = adj @ adj
        row, col, edge_weight = adj.coo()
        edge_index = torch.stack([row, col], dim=0)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight