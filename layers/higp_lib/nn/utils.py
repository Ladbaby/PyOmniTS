from typing import Optional
import torch
from torch import Tensor

def maybe_cat_emb(x: Tensor, emb: Optional[Tensor]):
    if emb is None:
        return x
    if emb.ndim < x.ndim:
        if emb.ndim == 3 and x.ndim == 4:
            emb = emb.unsqueeze(1)
        else:
            emb = emb[[None] * (x.ndim - emb.ndim)]
    emb = emb.expand(*x.shape[:-1], -1)
    return torch.cat([x, emb], dim=-1)
