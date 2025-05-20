r"""#TODO add module summary line.

#TODO add module description.
"""

__all__ = [
    # Classes
    "ScaledDotProductAttention",
]

from math import sqrt
from typing import Optional

import torch
from torch import Tensor, nn

from data.dependencies.tsdm.utils.decorators import autojit


@autojit
class ScaledDotProductAttention(nn.Module):
    r"""Permutation-invariant dot-product attention."""

    HP: dict = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__doc__": __doc__,
        "__module__": __module__,  # type: ignore[name-defined]
        "dim_k": int,
        "dim_v": int,
        "output_size": int,
        "num_heads": int,
    }
    r"""Dictionary of hyperparameters."""

    # BUFFERS
    scale: Tensor
    attention_weights: Tensor

    def __init__(
        self,
        dim_k: int,
        dim_v: int,
        output_size: int,
        *,
        num_heads: int = 4,
        dim_k_latent: Optional[int] = None,
        dim_v_latent: Optional[int] = None,
    ) -> None:
        super().__init__()
        dim_q = dim_k

        dim_k_latent = max(1, dim_k // 2) if dim_k_latent is None else dim_k_latent
        dim_v_latent = dim_v if dim_v_latent is None else dim_v_latent

        Wq = torch.zeros((num_heads, dim_k_latent))
        Wk = torch.randn((dim_k, num_heads, dim_k_latent)) / sqrt(dim_k)
        Wv = torch.randn((dim_v, num_heads, dim_v_latent)) / sqrt(dim_v)
        Wo = torch.randn((num_heads, dim_v_latent, output_size)) / sqrt(
            num_heads * dim_v_latent
        )

        self.Wq = nn.Parameter(Wq)
        self.Wk = nn.Parameter(Wk)
        self.Wv = nn.Parameter(Wv)
        self.Wo = nn.Parameter(Wo)
        # self.softmax = nn.Softmax(dim=-2)
        self.register_buffer("scale", torch.tensor(1 / sqrt(dim_q)))
        self.register_buffer("attention_weights", torch.tensor([]))

    def forward(self, K: Tensor, V: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        r""".. Signature:: ``(..., *L, d), (..., *L, e) -> (..., k)``.

        After a forward pass is performed, the attention weights can be
        accessed via the `attention_weights` buffer.

        - Q: `(h, dim_k)`
        - K: `(..., *L, dim_k)`
        - V: `(..., *L, dim_v)`
        """
        if mask is None:
            mask = torch.isnan(K[..., 0])

        Q = self.Wq
        K = torch.einsum("...d, dhk -> ...hk", K, self.Wk)
        V = torch.einsum("...e, ehv -> ...hv", V, self.Wv)
        QK = torch.einsum("hd, ...hd -> ...h", Q, K)
        QK[mask] = float("-inf")
        w = nn.functional.softmax(self.scale * QK, dim=-2)
        # w = self.softmax(self.scale * QK)
        self.attention_weights = w
        QKV = torch.nanmean(w[..., None] * V, dim=-3)  # ...h, ...Lhv -> ...hv
        return torch.einsum("...hv, hvr -> ...r", QKV, self.Wo)
