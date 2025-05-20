from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.typing import Adj

from .select import SelectOutput


class Connect(torch.nn.Module):
    r"""An abstract base class implementing custom edge connection operators.

    Specifically, :class:`Connect` determines for each pair of supernodes the
    presence or abscene of an edge based on the existing edges between the
    nodes in the two supernodes.
    The operator also computes new coarsened edge features (if present).
    """

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        pass

    def connect(self,
                edge_index: Adj,
                edge_attr: Optional[Tensor] = None,
                s: SelectOutput = None,
                *,
                batch: Optional[Tensor] = None,
                num_nodes: Optional[int] = None,
                **kwargs) -> Tuple[Adj, Optional[Tensor]]:
        r"""
        Args:
            edge_index (torch.Tensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
            s (SelectOutput): The mapping from nodes to supernodes.
                (default: :obj:`None`)
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific graph. (default: :obj:`None`)
            num_nodes (int, optional): The number of nodes.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
