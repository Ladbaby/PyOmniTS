import torch
from torch import nn
from torch_geometric.utils import to_dense_adj
from torch_sparse import SparseTensor

from layers.higp_lib.nn.hierarchical.pooling.mincut_pool import MinCutPool
import layers.tsl as tsl

from layers.tsl.utils import ensure_list

class MinCutHierarchyBuilder(nn.Module):
    r"""Hierarchy encoder"""

    def __init__(self,
                 n_nodes: int,
                 hidden_size: int,
                 n_clusters: float,
                 n_levels: int = 1,
                 temp_decay: float = 0.99995,
                 hard=True):
        super(MinCutHierarchyBuilder, self).__init__()

        self.n_levels = n_levels
        input_nodes = n_nodes
        pooling_layers = []
        n_clusters = ensure_list(n_clusters)
        if len(n_clusters) != n_levels - 2:
            assert len(n_clusters) == 1
            n_clusters = n_clusters * (n_levels - 2)
        for i in range(n_levels - 2):
            pooling_layers.append(MinCutPool(emb_size=hidden_size,
                                             n_nodes=input_nodes,
                                             n_clusters=n_clusters[i],
                                             hard=hard,
                                             temp_decay=temp_decay))
            input_nodes = n_clusters[i]

        self.pooling_layers = nn.ModuleList(pooling_layers)

    def forward(self, emb, edge_index, edge_weight=None):
        # emb: [nodes features]
        if isinstance(edge_index, SparseTensor):
            adj = edge_index.to_dense()
        else:
            adj = to_dense_adj(edge_index, edge_attr=edge_weight)[0].T

        # force the graph to be undirected
        adj = torch.max(adj, adj.T)

        d = torch.sum(adj, dim=-1, keepdim=True)
        d = 1 / (torch.sqrt(d) + tsl.epsilon)
        adj = d * adj * d.T

        embs = [emb]
        adjs = [adj]
        seletcs = [None]
        sizes = [emb.size(-2)]
        min_cut_loss = 0.
        reg_loss = 0.
        for i in range(self.n_levels - 2):
            # Pooling
            v, adj_, s, (mc_loss, r_loss) = \
                self.pooling_layers[i](embs[i], adjs[i])
            # Update embedding
            seletcs.append(s)
            embs.append(v)
            adjs.append(adj_)
            sizes.append(v.size(-2))
            min_cut_loss += mc_loss
            reg_loss += r_loss
        # add the last level
        embs.append(emb.mean(-2, keepdim=True))
        adjs.append(None)
        if self.n_levels > 2 and emb.dim() == 3:
            s_tot = torch.ones(emb.size(0), sizes[-1], 1, device=emb.device)
        else:
            s_tot = torch.ones(sizes[-1], 1, device=emb.device)
        seletcs.append(s_tot)
        sizes.append(1)
        return embs, adjs, seletcs, sizes, (min_cut_loss, reg_loss)
