from functools import partial

from torch import nn
import torch

from layers.higp_lib.nn.hierarchical.ops import src_reduce, src_lift
from layers.higp_lib.nn.layers import GraphAnisoConv
from layers.tsl.nn.blocks import ResidualMLP
from layers.tsl.nn.layers import GraphConv, DiffConv


class PyramidalGNN(nn.Module):
    r""""""

    def __init__(self,
                 input_size,
                 hidden_size,
                 levels,
                 layers,
                 mode,
                 activation='silu',
                 k=2):
        super(PyramidalGNN, self).__init__()
        self.input_size = input_size
        self.levels = levels
        self.layers = layers
        self.mode = mode

        def _get_update_layers(in_size, out_size):
            layers = []
            for l in range(self.levels):
                if 0 < l < self.levels - 1:
                    in_size_ = 3 * in_size
                else:
                    in_size_ = 2 * in_size

                layers.append(ResidualMLP(input_size=in_size_,
                                          hidden_size=out_size,
                                          activation=activation))
            return nn.ModuleList(layers)

        update_layers = []
        gnn_layers = []

        if mode == 'gated':
            mp_cls = GraphAnisoConv
        elif mode == 'gcn':
            mp_cls = partial(GraphConv, norm='sym')
        elif mode == 'diff':
            mp_cls = partial(DiffConv, k=k)
        else:
            raise NotImplementedError

        for i in range(layers):
            if i == 0:
                update_layers.append(_get_update_layers(input_size,
                                                        hidden_size))
                gnn_layers.append(mp_cls(
                    input_size,
                    input_size,
                    activation=activation
                )
                )
            else:
                update_layers.append(_get_update_layers(hidden_size,
                                                        hidden_size))
                gnn_layers.append(mp_cls(
                    hidden_size,
                    hidden_size,
                    activation=activation
                )
                )

        self.update_layers = nn.ModuleList(update_layers)
        self.gnn_layers = nn.ModuleList(gnn_layers)

    def _inter_level_propagate(self, xs, selects, layers):
        out = []
        for cur_level in range(self.levels):
            if cur_level == 0:
                out_ = torch.cat([
                    xs[cur_level],
                    src_lift(xs[cur_level + 1], selects[cur_level + 1])
                ], dim=-1)
            elif cur_level == self.levels - 1:
                out_ = torch.cat([
                    xs[cur_level],
                    src_reduce(xs[cur_level - 1], selects[cur_level])
                ], dim=-1)
            else:
                out_ = torch.cat([
                    xs[cur_level],
                    src_reduce(xs[cur_level - 1], selects[cur_level]),
                    src_lift(xs[cur_level + 1], selects[cur_level + 1])
                ], dim=-1)
            out.append(layers[cur_level](out_))
        return out

    def forward(self, xs, selects, edge_index, edge_weight=None):
        # x: List[[batches nodes features]]
        outs = xs
        if self.mode == 'gated':
            gnn_kwargs = {'edge_index': edge_index}
        else:
            gnn_kwargs = {'edge_index': edge_index,
                          'edge_weight': edge_weight}

        for cur_layer in range(self.layers):
            outs[0] = self.gnn_layers[cur_layer](outs[0], **gnn_kwargs)
            outs = self._inter_level_propagate(outs,
                                               selects,
                                               self.update_layers[cur_layer])
        return outs
