from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch_geometric.typing import Adj
from torch_sparse import SparseTensor


def connectivity_to_row_col(edge_index: Adj) -> Tuple[Tensor, Tensor]:
    if isinstance(edge_index, Tensor):
        return edge_index[0], edge_index[1]
    elif isinstance(edge_index, SparseTensor):
        row, col, _ = edge_index.t().coo()
        return row, col
    else:
        raise NotImplementedError()


def connectivity_to_edge_index(
        edge_index: Adj,
        edge_attr: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    if isinstance(edge_index, Tensor):
        return edge_index, edge_attr
    elif isinstance(edge_index, SparseTensor):
        row, col, edge_attr = edge_index.t().coo()
        edge_index = torch.stack([row, col], dim=0)
        return edge_index, edge_attr
    else:
        raise NotImplementedError()


def connectivity_to_adj_t(edge_index: Adj,
                          edge_attr: Optional[Tensor] = None,
                          num_nodes: Optional[int] = None) -> SparseTensor:
    if isinstance(edge_index, SparseTensor):
        return edge_index
    elif isinstance(edge_index, Tensor):
        adj_t = SparseTensor.from_edge_index(edge_index, edge_attr,
                                             (num_nodes, num_nodes)).t()
        return adj_t
    else:
        raise NotImplementedError()


def broadcast_shape(src: Tensor, shape: List, dim: int):
    size = [1] * len(shape)
    size[dim] = -1
    return src.view(size).expand(shape)


def expand(x_red: Tensor, assignment: Tensor, node_dim=-2):
    num_nodes = len(assignment)
    exp_shape = list(x_red.size())
    exp_shape[node_dim] = num_nodes

    red_index = broadcast_shape(assignment, exp_shape, node_dim)
    x = x_red.gather(node_dim, red_index)

    return x


def pseudo_inverse(
        edge_index: Adj,
        edge_attr: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
) -> Tuple[Adj, Optional[Tensor]]:
    if isinstance(edge_index, Tensor):
        adj = connectivity_to_adj_t(edge_index, edge_attr, num_nodes)
    elif isinstance(edge_index, SparseTensor):
        adj = edge_index
    else:
        raise NotImplementedError()
    adj_inv = torch.linalg.pinv(adj.to_dense())
    adj_inv[adj_inv < 1e-4] = 0  # clip small values due to numerical errors
    adj_inv = SparseTensor.from_dense(adj_inv)
    if isinstance(edge_index, Tensor):
        return connectivity_to_edge_index(adj_inv)
    else:
        return adj_inv, None


def make_graph_connected_(adj: np.ndarray,
                          similarity: np.ndarray,
                          min_val: float):
    r"""Makes the given graph connected by adding edges between the largest
    connected components.

    Args:
        adj (np.ndarray): The adjacency matrix of the graph.
        similarity (np.ndarray): The similarity matrix of the graph.
        min_val (float): The minimum value of the similarity matrix.
    """
    # Keep only the largest connected component
    import scipy.sparse as sp
    assert min_val <= adj[adj > 0].min()

    np.fill_diagonal(similarity, 0)
    sim = pd.DataFrame(similarity)

    num_components, component = sp.csgraph.connected_components(
        sp.csr_matrix(adj), connection='weak')

    sim.columns = pd.MultiIndex.from_tuples(zip(component, sim.columns))
    sim.index = sim.columns

    store = [dict(max_val=0, r=None, c=None) for _ in range(num_components)]
    for i in range(num_components - 1):
        group_sim = sim.loc[i]
        for j in range(i + 1, num_components):
            subset_sim = group_sim.loc[:, j]
            sub_values = subset_sim.values
            row, col = np.unravel_index(sub_values.argmax(), sub_values.shape)
            row, col = subset_sim.index[row], subset_sim.columns[col]
            val = subset_sim.loc[row, col]
            if val > store[i]['max_val']:
                store[i].update(max_val=val, r=row, c=col)
            if val > store[j]['max_val']:
                store[j].update(max_val=val, r=col, c=row)
    # Update the adjacency matrix
    for store in store:
        if store['max_val'] > 0:
            adj[store['r'], store['c']] = min_val
            adj[store['c'], store['r']] = min_val

    return adj
