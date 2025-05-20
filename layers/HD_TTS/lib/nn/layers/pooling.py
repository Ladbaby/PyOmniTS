from typing import Optional, Literal, List, Union, Tuple

import torch
from torch import nn, Tensor
from layers.tsl.utils import ensure_list

from layers.HD_TTS.lib.pooling import KMISPooling
from layers.HD_TTS.lib.pooling import connectivity_to_adj_t, Pooling, SelectOutput
from layers.HD_TTS.lib.pooling.aggr_pool import ReductionType, ConnectionType, LiftMatrixType
from .anisotropic import GraphAnisoConv
from .diff_conv import DiffConv
from .prop_conv import PropConv

MessagePassingMethods = Literal[
    "diffconv",
    "diffconv_sym",
    "anisoconv",
    "propconv",
]


class HierarchicalPooling(nn.Module):

    def __init__(self,
                 pool: Pooling,
                 mp: nn.ModuleList,
                 recursive_lifting: bool = True,
                 mp_stage: str = "both",
                 keep_initial_features: bool = True,
                 unpool_with_inverse: bool = True):
        super().__init__()

        self.pool = pool
        self.mps = mp

        assert mp_stage in ["pre", "post", "both"]
        self.mp_stage = mp_stage

        self.n_layers = len(self.mps)
        self.recursive_lifting = recursive_lifting
        self.keep_initial_features = keep_initial_features
        self.unpool_with_inverse = unpool_with_inverse

        self._graphs = None

    def compute_coarsened_graphs(self, edge_index: Tensor,
                                 edge_attr: Optional[Tensor] = None,
                                 num_nodes: int = None):
        out = []
        adj = connectivity_to_adj_t(edge_index, edge_attr, num_nodes=num_nodes)
        for _ in range(self.n_layers):
            # select
            s = self.pool.select(adj)
            # store initial graph
            s.adj = adj
            # lift
            if self.unpool_with_inverse:
                s.set_s_inv("inverse")  # otherwise defaults to transpose
            # connect
            adj, _ = self.pool.connect(adj, s=s)
            # skip-conn select from initial graph
            if len(out) and not self.recursive_lifting:
                s.s_inv = self.pool.lift(x_pool=s.s_inv.t(), s=out[-1]).t()
            # store pooled graph
            s.adj_pool = adj
            out.append(s)
        return out

    def get_coarsened_graphs(self, edge_index: Tensor = None,
                             edge_attr: Optional[Tensor] = None,
                             num_nodes: int = None,
                             cached: bool = False):
        # compute coarsened graphs
        if self._graphs is None:
            assert edge_index is not None
            graphs = self.compute_coarsened_graphs(edge_index, edge_attr,
                                                   num_nodes=num_nodes)
            if cached:
                self._graphs = graphs
        else:
            graphs = self._graphs
        return graphs

    def get_mps(self, mps):
        if self.mp_stage == "pre":
            return mps, None
        elif self.mp_stage == "post":
            return None, mps
        elif self.mp_stage == "both":
            return mps

    def mp_pool_lift_mp(self,
                        x: Tensor,
                        convs: nn.ModuleList,
                        graphs: list):
        if len(graphs):
            graph = graphs.pop(0)
            conv_h, conv_g = convs.pop(0)

            # mp1
            x = conv_h(x, graph.adj)
            # pool
            x_pool = self.pool.reduce(x, s=graph)

            # recursion
            x_pool = self.mp_pool_lift_mp(x_pool, convs, graphs)

            # lift
            x_lift = self.pool.lift(x_pool, s=graph)

            # mp2
            x = conv_g(x_lift, graph.adj)
            return x
        else:
            return x

    # Use for recursive lifting
    def lift(self, x: Tensor, convs: nn.ModuleList, graphs: list):
        graph = graphs.pop(-1)
        _, conv_g = convs.pop(-1)
        # lift
        x_lift = self.pool.lift(x, s=graph)
        # mp2
        x = conv_g(x_lift, graph.adj)
        # recursion
        if len(graphs):
            x = self.lift(x, convs, graphs)
        return x

    def recursive_lifting_pooling(self, x: Tensor, graphs: List[SelectOutput]) \
            -> Tuple[List[Tensor], List[Tensor]]:
        # Compute pooled features
        pooled = []
        for convs, graph in zip(self.mps, graphs):
            conv_h, _ = self.get_mps(convs)
            # 1st message-passing
            x = conv_h(x, graph.adj)
            # pooling
            x = self.pool.reduce(x, s=graph)
            pooled.append(x)

        # Recursively compute lifted features
        out = []
        for i in range(1, len(graphs) + 1):
            # x_out = self.mp_pool_lift_mp(x, self.mps[:i], graphs[:i])
            x_out = self.lift(pooled[i - 1], self.mps[:i], graphs[:i])
            out.append(x_out)

        return out, pooled

    def skip_lifting_pooling(self, x: Tensor, graphs: List[SelectOutput]) \
            -> Tuple[List[Tensor], List[Tensor]]:
        out, pooled = [], []
        # Skip-lifting strategy
        x_pool = x
        for convs, graph in zip(self.mps, graphs):
            conv_h, conv_g = self.get_mps(convs)
            if self.mp_stage in ["pre", "both"]:
                # 1st message-passing
                x_pool = conv_h(x_pool, graph.adj)
            # pooling
            x_pool = self.pool.reduce(x_pool, s=graph)
            # store pooled features
            pooled.append(x_pool)
            if self.mp_stage in ["post", "both"]:
                # 2nd message-passing
                x_pool = conv_g(x_pool, graph.adj_pool)
            # unpooling
            x_out = self.pool.lift(x_pool, s=graph)

            out.append(x_out)
        return out, pooled

    def forward(self, x: Tensor, edge_index: Tensor,
                edge_attr: Optional[Tensor] = None,
                cached: bool = False) -> Tuple[Tensor, List[Tensor]]:
        # x: [..., node, features]
        num_nodes = x.size(-2)
        graphs = self.get_coarsened_graphs(edge_index, edge_attr, num_nodes,
                                           cached=cached)

        out = [x] if self.keep_initial_features else []
        pooled = [x] if self.keep_initial_features else []

        if self.recursive_lifting:
            x_out, x_pooled = self.recursive_lifting_pooling(x, graphs)
        else:
            x_out, x_pooled = self.skip_lifting_pooling(x, graphs)

        out += x_out
        pooled += x_pooled

        # -> [batch, layers, ..., node, features]
        return torch.stack(out, dim=1), pooled


class HierPoolFactory(HierarchicalPooling):

    def __init__(self, input_size: int,
                 hidden_size: Union[int, List[int]],
                 n_layers: int,
                 # Pooling layers
                 reduce_op: ReductionType = "sum",
                 connect_op: ConnectionType = "mean",
                 lift_matrix: LiftMatrixType = "inverse",
                 # MP layers
                 mp_method: MessagePassingMethods = "diffconv",
                 kernel_size: int = 1,
                 activation: str = "relu",
                 mp_stage: str = "both",
                 # Hierarchical pooling params
                 recursive_lifting: bool = True,
                 keep_initial_features: bool = True):

        hidden_size = ensure_list(hidden_size)
        if keep_initial_features:
            assert hidden_size[0] == input_size

        #  POOLING LAYER  #####################################################
        pool = KMISPooling(reduce=reduce_op,
                           connect=connect_op,
                           scorer="constant",
                           remove_self_loops=True,
                           force_undirected=True)

        #  MESSAGE-PASSING LAYERS  ############################################
        mp_methods = ensure_list(mp_method)
        mp_class, mp_kwargs = [], []
        for mp_method in mp_methods:
            if mp_method == "diffconv" or mp_method == "diffconv_sym":
                mp_class.append(DiffConv)
                mp_kwargs.append(dict(k=kernel_size,
                                      root_weight=True,
                                      add_backward=mp_method == "diffconv",
                                      activation=activation,
                                      cached=True))
            elif mp_method == "anisoconv":
                mp_class.append(GraphAnisoConv)
                mp_kwargs.append(dict(kernel_size=kernel_size,
                                      activation=activation))
            elif mp_method == "propconv":
                mp_class.append(PropConv)
                mp_kwargs.append(dict(add_backward=True,
                                      use_edge_weights=True,
                                      normalize_weights=True))
            else:
                raise NotImplementedError()

        if mp_stage == "both":
            # allow for different hidden sizes in the pyramid
            if len(hidden_size) == 1:
                hidden_size = hidden_size * n_layers
            assert len(hidden_size) == n_layers
            if len(mp_class) == 1:
                mp_class = mp_class * 2
                mp_kwargs = mp_kwargs * 2

            mps = nn.ModuleList([
                nn.ModuleList([
                    mp_class[0](in_channels=(input_size if i == 0 else
                                             hidden_size[i - 1]),
                                out_channels=hidden_size[i],
                                **mp_kwargs[0]),
                    mp_class[1](in_channels=hidden_size[i],
                                out_channels=(hidden_size[0] if i == 0 else
                                              hidden_size[i - 1]),
                                **mp_kwargs[1])
                ]) for i in range(n_layers)
            ])
        else:
            assert len(hidden_size) == len(mp_class) == 1
            mp_class, mp_kwargs = mp_class[0], mp_kwargs[0]
            hidden_size = hidden_size[0]
            mps = nn.ModuleList([
                mp_class(in_channels=input_size if i == 0 else hidden_size,
                         out_channels=hidden_size,
                         **mp_kwargs)
                for i in range(n_layers)
            ])

        super().__init__(pool=pool,
                         mp=mps,
                         recursive_lifting=recursive_lifting,
                         mp_stage=mp_stage,
                         keep_initial_features=keep_initial_features,
                         unpool_with_inverse=lift_matrix == "inverse")
