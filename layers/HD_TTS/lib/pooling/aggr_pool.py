from typing import Literal, Optional, Tuple

import torch_sparse
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import scatter, coalesce, remove_self_loops as rsl
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import SparseTensor

from .base import Lift, Reduce, Connect, SelectOutput
from .utils import (pseudo_inverse,
                    connectivity_to_edge_index,
                    connectivity_to_adj_t)

LiftMatrixType = Literal["precomputed", "transpose", "inverse"]
ReductionType = Literal["sum", "mean", "min", "max"]
ConnectionType = Literal["sum", "mean", "min", "max", "mul", "add"]


#  AGGR REDUCE  ###############################################################
def reduce(x: Tensor, assignment: Tensor,
           num_nodes: int = None,
           reduce: ReductionType = "sum",
           node_dim: int = -2):
    assert reduce in ["sum", "mean", "min", "max"]
    num_nodes = maybe_num_nodes(assignment, num_nodes)
    return scatter(x, assignment, node_dim, num_nodes, reduce)


class AggrReduce(Reduce):
    r"""The reduction operator from SRC

    .. math::
        \mathrm{sum}(\mathcal{X}) = \sum_{\mathbf{x}_i \in \mathcal{X}}
        \mathbf{x}_i.
    """

    def __init__(self, operation: ReductionType = "sum"):
        super().__init__()
        self.operation = operation

    def forward(self,
                x: Tensor,
                s: SelectOutput = None,
                *,
                batch: Optional[Tensor] = None,
                num_nodes: Optional[int] = None,
                **kwargs) -> Tensor:
        return torch_sparse.matmul(s.s.t(), x, reduce=self.operation)


#  AGGR LIFT  #################################################################

class AggrLift(Lift):

    def __init__(self, matrix_op: LiftMatrixType = "precomputed",
                 operation: ReductionType = "sum"):
        super().__init__()
        matrix_op_types = list(LiftMatrixType.__args__)
        assert matrix_op in matrix_op_types, \
            f"'matrix_op' must be one of {matrix_op_types} ({matrix_op} given)"
        self.matrix_op = matrix_op
        self.operation = operation

    def forward(self,
                x_pool: Tensor,
                s: SelectOutput = None,
                *,
                batch: Optional[Tensor] = None,
                num_nodes: Optional[int] = None,
                **kwargs) -> Tensor:
        if self.matrix_op == "precomputed":
            s_inv = s.s_inv.t()
        elif self.matrix_op == "inverse":
            s_inv = pseudo_inverse(s.s)[0].t()
        elif self.matrix_op == "transpose":
            s_inv = s.s
        else:
            raise RuntimeError()
        x_prime = torch_sparse.matmul(s_inv, x_pool, reduce=self.operation)
        return x_prime


#  AGGR CONNECT  ##############################################################

def connect(assignment: Tensor, edge_index: Adj, edge_attr: OptTensor = None,
            num_nodes: int = None,
            remove_self_loops: bool = True,
            reduce: ConnectionType = "sum"):
    to_sparse = False
    if isinstance(edge_index, SparseTensor):
        edge_index, edge_attr = connectivity_to_edge_index(edge_index,
                                                           edge_attr)
        to_sparse = True
    edge_index = assignment[edge_index]
    edge_index, edge_attr = coalesce(edge_index, edge_attr,
                                     num_nodes=num_nodes, reduce=reduce)
    if remove_self_loops:
        edge_index, edge_attr = rsl(edge_index, edge_attr)
    if to_sparse:
        edge_index = connectivity_to_adj_t(edge_index, edge_attr, num_nodes)
        edge_attr = None
    return edge_index, edge_attr


class AggrConnect(Connect):
    r"""The connection operator from SRC

    .. math::
        \mathrm{sum}(\mathcal{X}) = \sum_{\mathbf{x}_i \in \mathcal{X}}
        \mathbf{x}_i.
    """

    def __init__(self, reduce: ConnectionType = "sum",
                 remove_self_loops: bool = True):
        super().__init__()
        self.reduce = reduce
        self.remove_self_loops = remove_self_loops

    def forward(self,
                edge_index: Adj,
                edge_attr: Optional[Tensor] = None,
                s: SelectOutput = None,
                *,
                batch: Optional[Tensor] = None,
                **kwargs) -> Tuple[Adj, Optional[Tensor]]:
        return connect(s.cluster_index, edge_index, edge_attr,
                       num_nodes=s.num_clusters,
                       remove_self_loops=self.remove_self_loops,
                       reduce=self.reduce)
