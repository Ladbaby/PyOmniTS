from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from layers.tsl.nn.layers import DiffConv as DiffConv_


class DiffConv(DiffConv_):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 k: int,
                 root_weight: bool = True,
                 add_backward: bool = True,
                 bias: bool = True,
                 activation: str = None,
                 cached: bool = False):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         k=k,
                         root_weight=root_weight,
                         add_backward=add_backward,
                         bias=bias,
                         activation=activation)
        self.cached = cached

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None, cache_support: bool = None) \
            -> Tensor:
        if cache_support is None:
            cache_support = self.cached
        return super().forward(x, edge_index, edge_weight, cache_support)
