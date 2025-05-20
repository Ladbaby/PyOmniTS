r"""#TODO add module summary line.

#TODO add module description.
"""

__all__ = [
    # Classes
    "DeepSet",
    "DeepSetReZero",
]

from typing import Optional

import torch
from torch import Tensor, nn

from data.dependencies.tsdm.models.generic.mlp import MLP
from data.dependencies.tsdm.models.generic.rezero import ReZeroMLP
from data.dependencies.tsdm.utils.decorators import autojit


@autojit
class DeepSet(nn.ModuleDict):
    r"""Permutation invariant deep set model."""

    HP: dict = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__doc__": __doc__,
        "__module__": __module__,  # type: ignore[name-defined]
        "input_size": int,
        "output_size": int,
        "latent_size": int,
        "bottleneck_size": int,
        "encoder": MLP.HP,
        "decoder": MLP.HP,
    }
    r"""Dictionary of hyperparameters."""

    def __init__(
        self,
        inputs_size: int,
        output_size: int,
        *,
        latent_size: Optional[int] = None,
        hidden_size: Optional[int] = None,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        # aggregation: Literal["min", "max", "sum", "mean", "prod"] = "sum",
    ):
        # super().__init__()
        latent_size = inputs_size if latent_size is None else latent_size
        hidden_size = inputs_size if hidden_size is None else hidden_size
        encoder = MLP(
            inputs_size, latent_size, hidden_size=hidden_size, num_layers=encoder_layers
        )
        decoder = MLP(
            latent_size, output_size, hidden_size=hidden_size, num_layers=decoder_layers
        )
        super().__init__({"encoder": encoder, "decoder": decoder})
        self.encoder = self["encoder"]
        self.decoder = self["decoder"]

    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., *V, D) -> (..., F)``.

        Components:
          - Encoder: `(..., D) -> (..., E)`
          - Aggregation: `(..., *V, E) -> (..., E)`
          - Decoder: `(..., E) -> (..., F)`

        Parameters
        ----------
        x: Tensor

        Returns
        -------
        Tensor
        """
        x = self.encoder(x)
        x = torch.nanmean(x, dim=-2)
        x = self.decoder(x)
        return x


@autojit
class DeepSetReZero(nn.ModuleDict):
    r"""Permutation invariant deep set model."""

    HP: dict = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__doc__": __doc__,
        "__module__": __module__,  # type: ignore[name-defined]
        "input_size": int,
        "output_size": int,
        "latent_size": int,
        "bottleneck_size": int,
        "encoder": MLP.HP,
        "decoder": MLP.HP,
    }
    r"""Dictionary of hyperparameters."""

    def __init__(
        self,
        inputs_size: int,
        output_size: int,
        *,
        latent_size: Optional[int] = None,
        hidden_size: Optional[int] = None,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        # aggregation: Literal["min", "max", "sum", "mean", "prod"] = "sum",
    ):
        # super().__init__()
        latent_size = inputs_size if latent_size is None else latent_size
        hidden_size = inputs_size if hidden_size is None else hidden_size
        encoder = ReZeroMLP(
            inputs_size, latent_size, latent_size=hidden_size, num_blocks=encoder_layers
        )
        decoder = ReZeroMLP(
            latent_size, output_size, latent_size=hidden_size, num_blocks=decoder_layers
        )
        super().__init__({"encoder": encoder, "decoder": decoder})
        self.encoder = self["encoder"]
        self.decoder = self["decoder"]

    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature:: `(..., *V, D) -> (..., F)`.

        Components:
          - Encoder: ``(..., D) -> (..., E)``.
          - Aggregation: ``(..., *V, E) -> (..., E)``.
          - Decoder: ``(..., E) -> (..., F)``.

        Parameters
        ----------
        x: Tensor

        Returns
        -------
        Tensor
        """
        x = self.encoder(x)
        x = torch.nanmean(x, dim=-2)
        x = self.decoder(x)
        return x
