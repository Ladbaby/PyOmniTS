import torch
from torch import nn

from layers.higp_lib.nn.hierarchical.ops import src_reduce, compute_aggregation_matrix
from layers.higp_lib.nn.utils import maybe_cat_emb
from layers.tsl.nn import maybe_cat_exog


class HierarchyEncoder(nn.Module):
    r"""Hierarchy Builder"""
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 emb_size: int,
                 exog_size: int = 0):
        super(HierarchyEncoder, self).__init__()

        self.input_size = input_size

        input_size += exog_size + emb_size
        self.input_encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
        )

    def forward(self, x, u, embs, selects, cat_output=False):
        # cat exogenous then build aggregates
        x = maybe_cat_exog(x, u)
        C = compute_aggregation_matrix(selects, bottom_up=False)

        # compute aggregates
        xs = src_reduce(x, C.transpose(-1, -2))
        sizes = [s.size(-1) for s in selects[1:]]
        xs = list(torch.split(xs, sizes, dim=-2))
        xs = [x, ] + xs
        sizes = [x.size(-2), ] + sizes

        # cat embeddings
        xs = [maybe_cat_emb(x, emb) for x, emb in zip(xs, embs)]

        # encode
        xs = torch.cat(xs, dim=-2)
        xs = self.input_encoder(xs)
        if not cat_output:
            xs = torch.split(xs, sizes, dim=-2)
        return xs

