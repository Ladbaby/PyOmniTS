from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.typing import Adj
from torch_geometric.utils.mixin import CastMixin

from .connect import Connect
from .lift import Lift
from .reduce import Reduce
from .select import Select, SelectOutput


@dataclass
class PoolingOutput(CastMixin):
    r"""The pooling output of a :class:`torch_geometric.nn.pool.Pooling`
    module.

    Args:
        x (torch.Tensor): The pooled node features.
        edge_index (torch.Tensor): The coarsened edge indices.
        edge_attr (torch.Tensor, optional): The edge features of the coarsened
            graph. (default: :obj:`None`)
        batch (torch.Tensor, optional): The batch vector of the pooled nodes.
    """
    x: Tensor
    edge_index: Tensor
    edge_attr: Optional[Tensor] = None
    batch: Optional[Tensor] = None


class Pooling(torch.nn.Module):
    r"""A base class for pooling layers based on the
    `"Understanding Pooling in Graph Neural Networks"
    <https://arxiv.org/abs/1905.05178>`_ paper.

    :class:`Pooling` decomposes a pooling layer into three components:

    #. :class:`Select` defines how input nodes map to supernodes.

    #. :class:`Reduce` defines how input node features are aggregated.

    #. :class:`Lift` defines how pooled node features are un-pooled.

    #. :class:`Connect` decides how the supernodes are connected to each other.

    Args:
        selector (Select): The node selection operator.
        reducer (Reduce): The node feature aggregation operator.
        connector (Connect): The edge connection operator.
    """

    def __init__(self,
                 selector: Select = None,
                 reducer: Reduce = None,
                 lifter: Lift = None,
                 connector: Connect = None,
                 cached: bool = False,
                 node_dim: int = -2):
        super().__init__()
        self.selector = selector
        self.reducer = reducer
        self.lifter = lifter
        self.connector = connector
        self.cached = cached
        self.node_dim = node_dim
        self._s_cached = None

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.selector.reset_parameters()
        self.reducer.reset_parameters()
        self.lifter.reset_parameters()
        self.connector.reset_parameters()

    def select(self,
               edge_index: Adj,
               edge_attr: Optional[Tensor] = None,
               x: Optional[Tensor] = None,
               *,
               batch: Optional[Tensor] = None,
               num_nodes: Optional[int] = None,
               **kwargs) -> SelectOutput:
        r"""Implement the Select operation.

        Returns:
            The supernode assignment matrix
            :math:`\mathbf{S} \in \mathbb{R}^{N \times K}`.
        """
        if self.selector is not None:
            s = self.selector(edge_index=edge_index,
                              edge_attr=edge_attr,
                              x=x,
                              batch=batch,
                              num_nodes=num_nodes,
                              **kwargs)
            if self.cached:
                self._s_cached = s
            return s
        raise NotImplementedError

    def reduce(self,
               x: Tensor,
               s: SelectOutput = None,
               *,
               batch: Optional[Tensor] = None,
               num_nodes: Optional[int] = None,
               **kwargs) -> Tensor:
        r"""Implement the Reduce operation.

        Returns:
            The pooled supernode features :math:`\mathbf{X}_{pool}`.
        """
        if self.reducer is not None:
            if s is None and self.cached:
                s = self._s_cached
            return self.reducer(x=x, s=s,
                                batch=batch,
                                num_nodes=num_nodes,
                                **kwargs)
        raise NotImplementedError

    def lift(self,
             x_pool: Tensor,
             s: SelectOutput = None,
             *,
             batch: Optional[Tensor] = None,
             num_nodes: Optional[int] = None,
             **kwargs) -> Adj:
        """Implement the Lift operation.

        Returns:
            The un-pooled node features :math:`\mathbf{X}\prime`.
        """
        if self.lifter is not None:
            if s is None and self.cached:
                s = self._s_cached
            return self.lifter(x_pool=x_pool, s=s,
                               batch=batch,
                               num_nodes=num_nodes,
                               **kwargs)
        raise NotImplementedError

    def connect(self,
                edge_index: Adj,
                edge_attr: Optional[Tensor] = None,
                s: SelectOutput = None,
                *,
                batch: Optional[Tensor] = None,
                **kwargs) -> Tuple[Adj, Optional[Tensor]]:
        """Implement the Connect operation.

        Returns:
            The adjacency matrix of the coarse graph :math:`\mathbf{A}_{pool}`.
        """
        if self.connector is not None:
            if s is None and self.cached:
                s = self._s_cached
            return self.connector(edge_index=edge_index,
                                  edge_attr=edge_attr,
                                  s=s,
                                  batch=batch,
                                  **kwargs)
        raise NotImplementedError

    def clear_cache(self):
        self._s_cached = None

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(\n'
                f'  select={self.selector},\n'
                f'  reduce={self.reducer},\n'
                f'  lift={self.lifter},\n'
                f'  connect={self.connector},\n'
                f')')
