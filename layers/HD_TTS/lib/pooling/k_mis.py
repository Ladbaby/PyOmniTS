"""
    Maximal k-Independent Set (k-MIS) Pooling operator from
    `"Generalizing Downsampling from Regular Data to Graphs"
    <https://arxiv.org/abs/2208.03523>`_.

    Code adapted from https://github.com/pyg-team/pytorch_geometric/pull/6488
    by Francesco Landolfi (https://github.com/flandolfi)
"""

from typing import Callable, Optional, Tuple, Union

import torch
from torch_geometric.nn.dense import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor, Tensor
from torch_geometric.utils import scatter
from torch_geometric.utils import to_undirected, add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import SparseTensor

from .aggr_pool import (AggrReduce, AggrLift, AggrConnect,
                        ReductionType, ConnectionType)
from .base import Select, Pooling, PoolingOutput, SelectOutput
from .utils import (connectivity_to_row_col,
                    connectivity_to_edge_index,
                    connectivity_to_adj_t, broadcast_shape)

Scorer = Callable[[Tensor, Adj, OptTensor, OptTensor], Tensor]


def maximal_independent_set(edge_index: Adj, k: int = 1,
                            perm: OptTensor = None,
                            num_nodes: int = None) -> Tensor:
    r"""Returns a Maximal :math:`k`-Independent Set of a graph, i.e., a set of
    nodes (as a :class:`ByteTensor`) such that none of them are :math:`k`-hop
    neighbors, and any node in the graph has a :math:`k`-hop neighbor in the
    returned set.
    The algorithm greedily selects the nodes in their canonical order. If a
    permutation :obj:`perm` is provided, the nodes are extracted following
    that permutation instead.
    This method follows `Blelloch's Alogirithm
    <https://arxiv.org/abs/1202.3205>`_ for :math:`k = 1`, and its
    generalization by `Bacciu et al. <https://arxiv.org/abs/2208.03523>`_ for
    higher values of :math:`k`.
    Args:
        edge_index (Tensor or SparseTensor): The graph connectivity.
        k (int): The :math:`k` value (defaults to 1).
        perm (LongTensor, optional): Permutation vector. Must be of size
            :obj:`(n,)` (defaults to :obj:`None`).
    :rtype: :class:`ByteTensor`
    """
    n = maybe_num_nodes(edge_index.size(0), num_nodes)

    row, col = connectivity_to_row_col(edge_index)
    device = row.device

    # v2: ADD SELF-LOOPS
    row, col = add_remaining_self_loops(torch.stack([row, col]))[0]

    if perm is None:
        rank = torch.arange(n, dtype=torch.long, device=device)
    else:
        rank = torch.zeros_like(perm)
        rank[perm] = torch.arange(n, dtype=torch.long, device=device)

    mis = torch.zeros(n, dtype=torch.bool, device=device)
    mask = mis.clone()
    min_rank = rank.clone()

    while not mask.all():
        for _ in range(k):
            min_rank = scatter(min_rank[col], row, dim_size=n, reduce='min')

        mis = mis | torch.eq(rank, min_rank)
        mask = mis.clone().byte()

        for _ in range(k):
            mask = scatter(mask[row], col, dim_size=n, reduce='max')

        mask = mask.to(dtype=torch.bool)
        min_rank = rank.clone()
        min_rank[mask] = n

    return mis


def maximal_independent_set_cluster(edge_index: Adj, k: int = 1,
                                    perm: OptTensor = None,
                                    num_nodes: int = None) -> PairTensor:
    r"""Computes the Maximal :math:`k`-Independent Set (:math:`k`-MIS)
    clustering of a graph, as defined in `"Generalizing Downsampling from
    Regular Data to Graphs" <https://arxiv.org/abs/2208.03523>`_.
    The algorithm greedily selects the nodes in their canonical order. If a
    permutation :obj:`perm` is provided, the nodes are extracted following
    that permutation instead.
    This method returns both the :math:`k`-MIS and the clustering, where the
    :math:`c`-th cluster refers to the :math:`c`-th element of the
    :math:`k`-MIS.
    Args:
        edge_index (Tensor or SparseTensor): The graph connectivity.
        k (int): The :math:`k` value (defaults to 1).
        perm (LongTensor, optional): Permutation vector. Must be of size
            :obj:`(n,)` (defaults to :obj:`None`).
    :rtype: (:class:`ByteTensor`, :class:`LongTensor`)
    """
    n = maybe_num_nodes(edge_index, num_nodes)
    mis = maximal_independent_set(edge_index=edge_index, k=k, perm=perm,
                                  num_nodes=n)
    device = mis.device

    row, col = connectivity_to_row_col(edge_index)

    # v2: ADD SELF-LOOPS
    row, col = add_remaining_self_loops(torch.stack([row, col]))[0]

    if perm is None:
        rank = torch.arange(n, dtype=torch.long, device=device)
    else:
        rank = torch.zeros_like(perm)
        rank[perm] = torch.arange(n, dtype=torch.long, device=device)

    min_rank = torch.full((n,), fill_value=n, dtype=torch.long, device=device)
    rank_mis = rank[mis]
    min_rank[mis] = rank_mis

    for _ in range(k):
        min_rank = scatter(min_rank[row], col, dim_size=n, reduce='min')

    _, clusters = torch.unique(min_rank, return_inverse=True)
    perm = torch.argsort(rank_mis)

    # return mis, perm[clusters]

    _, clusters = torch.unique(perm[clusters], return_inverse=True)
    return mis, clusters


class KMISSelect(Select):
    _heuristics = {None, 'greedy', 'w-greedy'}
    _scorers = {
        'linear',
        'random',
        'constant',
        'canonical',
        'first',
        'last',
    }

    def __init__(self, in_channels: Optional[int] = None, k: int = 1,
                 scorer: Union[Scorer, str] = "constant",
                 score_heuristic: Optional[str] = 'greedy',
                 node_dim: int = -2,
                 force_undirected: bool = False):
        super(KMISSelect, self).__init__()
        assert score_heuristic in self._heuristics, \
            "Unrecognized `score_heuristic` value."

        if not callable(scorer):
            assert scorer in self._scorers, \
                "Unrecognized `scorer` value."

        self.k = k
        self.scorer = scorer
        self.score_heuristic = score_heuristic
        self.node_dim = node_dim
        self.force_undirected = force_undirected

        if scorer == 'linear':
            self.lin = Linear(in_channels=in_channels, out_channels=1,
                              weight_initializer='uniform')

    def _apply_heuristic(self, x: Tensor, adj: SparseTensor) -> Tensor:
        if self.score_heuristic is None:
            return x

        row, col = connectivity_to_row_col(adj)
        x = x.view(-1)

        if self.score_heuristic == 'greedy':
            k_sums = torch.ones_like(x)
        else:
            k_sums = x.clone()

        for _ in range(self.k):
            k_sums += scatter(k_sums[row], col, dim_size=adj.size(0),
                             reduce='add')

        return x / k_sums

    def _scorer(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None,
                batch: OptTensor = None,
                num_nodes: Optional[int] = None) -> Tensor:
        if self.scorer == 'linear':
            return self.lin(x).sigmoid()

        if isinstance(edge_index, SparseTensor):
            device = edge_index.device()
        else:
            device = edge_index.device

        if num_nodes is None:
            if isinstance(edge_index, SparseTensor):
                num_nodes = edge_index.size(0)
            else:
                assert x is not None
                num_nodes = x.size(self.node_dim)

        if self.scorer == 'random':
            return torch.rand((num_nodes, 1), device=device)

        if self.scorer == 'constant':
            return torch.ones((num_nodes, 1), device=device)

        if self.scorer == 'canonical':
            return -torch.arange(num_nodes, device=device).view(-1, 1)

        if self.scorer == 'first':
            return x[..., [0]]

        if self.scorer == 'last':
            return x[..., [-1]]

        return self.scorer(x, edge_index, edge_attr, batch)

    def forward(
            self,
            edge_index: Tensor,
            edge_attr: Optional[Tensor] = None,
            x: Optional[Tensor] = None,
            *,
            batch: Optional[Tensor] = None,
            num_nodes: Optional[int] = None,
            return_score: bool = False,
    ) -> Union[SelectOutput, Tuple[Tensor, Tensor, Tensor]]:
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
        if self.force_undirected:
            if isinstance(edge_index, SparseTensor):
                edge_index, edge_attr = connectivity_to_edge_index(edge_index)
            edge_index, edge_attr = to_undirected(edge_index, edge_attr,
                                                  num_nodes, reduce='max')
        adj = connectivity_to_adj_t(edge_index, edge_attr, num_nodes=num_nodes)
        score = self._scorer(x, adj, batch=batch, num_nodes=num_nodes)
        updated_score = self._apply_heuristic(score, adj)
        perm = torch.argsort(updated_score.view(-1), 0, descending=True)

        mis, cluster = maximal_independent_set_cluster(adj, self.k, perm)
        mis = mis.nonzero().view(-1)

        if return_score:
            return cluster, mis, score

        return SelectOutput(cluster_index=cluster)


class KMISPooling(Pooling):
    r"""Maximal :math:`k`-Independent Set (:math:`k`-MIS) pooling operator
    from `"Generalizing Downsampling from Regular Data to Graphs"
    <https://arxiv.org/abs/2208.03523>`_.
    Args:
        in_channels (int, optional): Size of each input sample. Ignored if
            :obj:`scorer` is not :obj:`"linear"`.
        k (int): The :math:`k` value (defaults to 1).
        scorer (str or Callable): A function that computes a score for every
            node. Nodes with higher score will have higher chances to be
            selected by the pooling algorithm. It can be one of the following:
            - :obj:`"linear"` (default): uses a sigmoid-activated linear
              layer to compute the scores
                .. note::
                    :obj:`in_channels` and :obj:`score_passthrough`
                    must be set when using this option.
            - :obj:`"random"`: assigns a score to every node extracted
              uniformly at random from 0 to 1;
            - :obj:`"constant"`: assigns :math:`1` to every node;
            - :obj:`"canonical"`: assigns :math:`-i` to every :math:`i`-th
              node;
            - :obj:`"first"` (or :obj:`"last"`): use the first (resp. last)
              feature dimension of :math:`\mathbf{X}` as node scores;
            - A custom function having as arguments
              :obj:`(x, edge_index, edge_attr, batch)`. It must return a
              one-dimensional :class:`FloatTensor`.
        score_heuristic (str, optional): Apply one of the following heuristic
            to increase the total score of the selected nodes. Given an
            initial score vector :math:`\mathbf{s} \in \mathbb{R}^n`,
            - :obj:`None`: no heuristic is applied;
            - :obj:`"greedy"` (default): compute the updated score
              :math:`\mathbf{s}'` as
                .. math::
                    \mathbf{s}' = \mathbf{s} \oslash (\mathbf{A} +
                    \mathbf{I})^k\mathbf{1},
              where :math:`\oslash` is the element-wise division;
            - :obj:`"w-greedy"`: compute the updated score
              :math:`\mathbf{s}'` as
                .. math::
                    \mathbf{s}' = \mathbf{s} \oslash (\mathbf{A} +
                    \mathbf{I})^k\mathbf{s},
              where :math:`\oslash` is the element-wise division. All scores
              must be strictly positive when using this option.
        score_passthrough (str, optional): Whether to aggregate the node
            scores to the feature vectors, using the function specified by
            :obj:`aggr_score`. If :obj:`"before"`, all the node scores are
            aggregated to their respective feature vector before the cluster
            aggregation. If :obj:`"after"`, the score of the selected nodes are
            aggregated to the feature vectors after the cluster aggregation.
            If :obj:`None`, the score is not aggregated. Defaults to
            :obj:`"before"`.
                .. note::
                    Set this option either to :obj:`"before"` or :obj:`"after"`
                    whenever :obj:`scorer` is :obj:`"linear"` or a
                    :class:`torch.nn.Module`, to make the scoring function
                    end-to-end differentiable.
        reduce (str or Aggregation, optional): The aggregation function to be
            applied to the nodes in the same cluster. Can be any string
            admitted by :obj:`scatter` (e.g., :obj:`'sum'`, :obj:`'mean'`,
            :obj:`'max'`) or any :class:`Aggregation`.
        connect (str): The aggregation function to be applied to the edges
            crossing the same two clusters. Can be any string
            admitted by :obj:`scatter` (e.g., :obj:`'sum'`, :obj:`'mean'`,
            :obj:`'max'`). Defaults to :obj:`'sum'`.
        remove_self_loops (bool): Whether to remove the self-loops from the
            graph after its reduction. Defaults to :obj:`True`.
    """
    _passthroughs = {None, 'before', 'after'}

    def __init__(self, in_channels: Optional[int] = None, k: int = 1,
                 scorer: Union[Scorer, str] = "constant",
                 score_heuristic: Optional[str] = "greedy",
                 score_passthrough: Optional[str] = "before",
                 reduce: ReductionType = "sum",
                 connect: ConnectionType = "sum",
                 remove_self_loops: bool = True,
                 force_undirected: bool = False) -> None:
        select = KMISSelect(in_channels=in_channels,
                            k=k,
                            scorer=scorer,
                            score_heuristic=score_heuristic,
                            force_undirected=force_undirected)

        assert score_passthrough in self._passthroughs, \
            "Unrecognized `score_passthrough` value."

        if scorer == 'linear':
            assert self.score_passthrough is not None, \
                "`'score_passthrough'` must not be `None`" \
                " when using `'linear'` scorer"

        self.score_passthrough = score_passthrough

        if reduce is not None:
            reduce = AggrReduce(operation=reduce)
        lift = AggrLift()
        connect = AggrConnect(reduce=connect,
                              remove_self_loops=remove_self_loops)

        super(KMISPooling, self).__init__(selector=select,
                                          reducer=reduce,
                                          lifter=lift,
                                          connector=connect)

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                *,
                batch: Optional[Tensor] = None,
                num_nodes: Optional[int] = None) -> PoolingOutput:
        """"""
        # Select
        cluster, mis, score = self.select(edge_index, edge_attr, x,
                                          batch=batch, num_nodes=num_nodes,
                                          return_score=True)

        # Reduce
        if self.score_passthrough == 'before':
            score = broadcast_shape(score, x.size(), dim=self.node_dim)
            x = x * score

        num_clusters = mis.size(0)

        if self.reducer is None:
            x = torch.index_select(x, dim=self.node_dim, index=mis)
        else:
            x = self.reduce(x, cluster, dim_size=num_clusters,
                            dim=self.node_dim)

        if self.score_passthrough == 'after':
            score = broadcast_shape(score[mis], x.size(), dim=self.node_dim)
            x = x * score

        # Connect
        edge_index, edge_attr = self.connect(cluster, edge_index, edge_attr,
                                             batch=batch)

        if batch is not None:
            batch = batch[mis]

        return PoolingOutput(x, edge_index, edge_attr, batch)

    def __repr__(self):
        if self.scorer == 'linear':
            channels = f"in_channels={self.lin.in_channels}, "
        else:
            channels = ""

        return f'{self.__class__.__name__}({channels}k={self.k})'
