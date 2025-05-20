import math

import torch
from einops import repeat
from torch import nn
from torch.nn import functional as F

import layers.tsl as tsl
from layers.higp_lib.nn.hierarchical.ops import src_reduce, src_connect
from layers.tsl.nn.layers.base import NodeEmbedding

def _rank3_trace(x):
    return torch.einsum('...jj->...', x)

class MinCutPool(nn.Module):
    def __init__(self,
                 emb_size: int,
                 n_nodes,
                 n_clusters,
                 hard=True,
                 temp: float = 1.0,
                 temp_decay: float = 0.99995):
        super(MinCutPool, self).__init__()
        self.in_channels = emb_size
        self.n_clusters = n_clusters
        self._temp = temp
        self.temp_decay = temp_decay
        self.hard = hard

        self.assigment_logits = NodeEmbedding(n_nodes=n_nodes, emb_size=n_clusters)

    def get_temp(self):
        if self.training:
            self._temp = max(self._temp * self.temp_decay, 0.05)
        return self._temp

    def compute_regularizations(self, s_soft, adj):
        """
        Compute MinCut and Orthogonality regularizations with soft assignment.
        """
        # MinCut regularization with soft assignment.
        # assert adj.dim() == 2, 'Adjacency matrix must be 2D'

        mincut_num = _rank3_trace(src_connect(adj, s_soft))
        d_flat = torch.sum(adj, dim=-1)
        d = torch.diag_embed(d_flat)
        mincut_den = _rank3_trace(src_connect(d, s_soft))
        mincut_loss = -(mincut_num / mincut_den)
        mincut_loss = torch.mean(mincut_loss)

        # Orthogonality regularization with soft assignment.
        ss = torch.matmul(s_soft.transpose(-2, -1), s_soft)
        i_s = torch.eye(self.n_clusters).type_as(ss)
        ortho_loss = torch.norm(
            ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
            i_s / math.sqrt(self.n_clusters), dim=(-1, -2))
        ortho_loss = torch.mean(ortho_loss)
        return mincut_loss, ortho_loss

    def forward(self, emb, adj):
        logits = self.assigment_logits()

        if emb.dim() == 3:
            logits = repeat(logits, 'n k -> b n k', b=emb.size(0))

        if self.training:
            s_sample = F.gumbel_softmax(logits / self.get_temp(),
                                      hard=self.hard,
                                      dim=-1)
        else:
            if self.hard:
                s_sample = F.one_hot(torch.argmax(logits, dim=-1),
                                     num_classes=self.n_clusters).float()
            else:
                s_sample = F.softmax(logits / self.get_temp(), dim=-1)

        s_sample += tsl.epsilon
        s_soft = F.softmax(self.assigment_logits() / self.get_temp(), dim=-1)

        if self.training:
            min_cut_loss, ortho_loss = self.compute_regularizations(s_soft, adj)
        else:
            min_cut_loss = ortho_loss = 0.

        # Compute the new node embeddings
        out_emb = src_reduce(emb, s_sample)
        # Compute the coarsened adjacency matrix
        out_adj = src_connect(adj, s_sample)

        # Fix and normalize coarsened adjacency matrix.
        ind = torch.arange(self.n_clusters, device=out_adj.device)
        out_adj[..., ind, ind] = 0.
        d = out_adj.sum(dim=-1, keepdim=True)
        d = torch.sqrt(d) + tsl.epsilon
        out_adj = (out_adj / d) / d.transpose(-2, -1)

        return out_emb, out_adj, s_sample, (min_cut_loss, ortho_loss)
