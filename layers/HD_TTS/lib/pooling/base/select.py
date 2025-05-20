from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
from torch_geometric.typing import Adj
from torch_sparse import SparseTensor

from layers.HD_TTS.lib.pooling.utils import pseudo_inverse


def cluster_to_s(cluster_index: Tensor,
                 node_index: Optional[Tensor] = None,
                 weight: Optional[Tensor] = None,
                 as_edge_index: bool = False,
                 num_nodes: Optional[int] = None,
                 num_clusters: Optional[int] = None):
    if num_nodes is None:
        num_nodes = cluster_index.size(0)
    if node_index is None:
        node_index = torch.arange(num_nodes, dtype=torch.long,
                                  device=cluster_index.device)
    if as_edge_index:
        return torch.stack([node_index, cluster_index], dim=0), weight
    else:
        return SparseTensor(row=node_index, col=cluster_index, value=weight,
                            sparse_sizes=(num_nodes, num_clusters))


# @torch.jit.script
@dataclass(init=False)
class SelectOutput:
    r"""The output of the :class:`Select` method, which holds an assignment
    from selected nodes to their respective cluster(s).

    Args:
        node_index (torch.Tensor): The indices of the selected nodes.
        num_nodes (int): The number of nodes.
        cluster_index (torch.Tensor): The indices of the clusters each node in
            :obj:`node_index` is assigned to.
        num_clusters (int): The number of clusters.
        weight (torch.Tensor, optional): A weight vector, denoting the strength
            of the assignment of a node to its cluster. (default: :obj:`None`)
    """
    s: SparseTensor
    node_index: Tensor
    num_nodes: int
    cluster_index: Tensor
    num_clusters: int
    weight: Optional[Tensor] = None
    s_inv: SparseTensor = None

    def __init__(
            self,
            s: SparseTensor = None,
            node_index: Tensor = None,
            num_nodes: int = None,
            cluster_index: Tensor = None,
            num_clusters: int = None,
            weight: Optional[Tensor] = None,
    ):
        if s is None:
            assert cluster_index is not None, \
                "'cluster_index' cannot be None if 's' is None"

            s = cluster_to_s(cluster_index,
                             node_index=node_index,
                             num_clusters=num_clusters,
                             num_nodes=num_nodes,
                             weight=weight)

        if node_index is None:
            node_index = s.coo()[0]

        if num_nodes is None:
            num_nodes = s.size(0)

        if cluster_index is None:
            cluster_index = s.coo()[1]

        if num_clusters is None:
            num_clusters = s.size(1)

        if weight is None:
            weight = s.coo()[2]

        self.s = s
        self.node_index = node_index
        self.num_nodes = num_nodes
        self.cluster_index = cluster_index
        self.num_clusters = num_clusters
        self.weight = weight
        self.s_inv = self.s.t()

    def set_s_inv(self, method):
        if method == "transpose":
            self.s_inv = self.s.t()
        elif method == "inverse":
            self.s_inv = pseudo_inverse(self.s)[0]
        else:
            raise ValueError()


class Select(torch.nn.Module):
    r"""An abstract base class implementing custom node selections that map the
    nodes of an input graph to supernodes of the pooled one.

    Specifically, :class:`Select` returns the supernode assignment matrix
    :math:`\mathbf{S} \in \mathbb{R}^{N \times K}`.
    """

    def reset_parameters(self):
        pass

    def forward(self,
                edge_index: Adj,
                edge_attr: Optional[Tensor] = None,
                x: Optional[Tensor] = None,
                *,
                batch: Optional[Tensor] = None,
                num_nodes: Optional[int] = None,
                **kwargs) -> SelectOutput:
        r"""
        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific graph. (default: :obj:`None`)
            num_nodes (int, optional): The number of nodes.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
