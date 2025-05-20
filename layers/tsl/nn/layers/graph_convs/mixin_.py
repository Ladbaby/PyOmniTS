import math
from typing import *

import torch
from torch import Tensor
from torch_sparse import SparseTensor

from layers.tsl.ops.connectivity import normalize_connectivity


class NormalizedAdjacencyMixin:
    r"""
    Mixin for layers which use a normalized adjiacency matrix to propagate messages
    """
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]] = None
    cached: bool = False
    normalization: str = 'none'

    def normalize_edge_index(self,
                             x,
                             edge_index,
                             edge_weight,
                             use_cached):
        if use_cached:
            if self._cached_edge_index is None:
                return self.normalize_edge_index(x, edge_index, edge_weight, False)
            return self._cached_edge_index
        if self.normalization == 'rw':
            edge_index, edge_weight = normalize_connectivity(edge_index,
                                                             edge_weight,
                                                             symmetric=True,
                                                             add_self_loops=False,
                                                             num_nodes=x.size(-2))
        elif self.normalization == 'sym':
            edge_index, edge_weight = normalize_connectivity(edge_index,
                                                             edge_weight,
                                                             symmetric=False,
                                                             add_self_loops=True,
                                                             num_nodes=x.size(-2))
        elif self.normalization == 'none':
            if edge_weight is None and not isinstance(edge_index, SparseTensor):
                edge_weight = torch.ones((edge_index.size(0),), device=edge_index.device, dtype=torch.int)
        else:
            raise NotImplementedError(f'Normalization must be one of: `rw`, `sym`, `none`')
        if self.cached:
            self._cached_edge_index = (edge_index, edge_weight)
        return edge_index, edge_weight
