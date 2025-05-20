r"""Encoders that work with torch tensors."""


__all__ = [
    # Classes
    "Time2Vec",
    "PositionalEncoder",
]

from typing import Final

import torch
from torch import Tensor, jit, nn

from data.dependencies.tsdm.utils.decorators import autojit


@autojit
class Time2Vec(nn.Module):
    r"""Learnable Time Encoding.

    References
    ----------
      - | Time2Vec: Learning a Vector Representation of Time
        | Seyed Mehran Kazemi, Rishab Goel, Sepehr Eghbali, Janahan Ramanan, Jaspreet
        | Sahota, Sanjay Thakur, Stella Wu, Cathal Smyth, Pascal Poupart, Marcus Brubaker
        | https: // arxiv.org / abs / 1907.05321
    """

    # Constants
    num_dim: Final[int]
    r"""Number of dimensions of the time encoding."""

    # Parameters
    freq: Tensor
    r"""Frequency of the time encoding."""
    phase: Tensor
    r"""Phase of the time encoding."""

    def __init__(self, num_dim: int, act: str = "sin") -> None:
        super().__init__()

        self.num_dim = num_dim
        self.freq = nn.Parameter(torch.randn(num_dim - 1))
        self.phase = nn.Parameter(torch.randn(num_dim - 1))

        if act == "sin":
            self.act = torch.sin
        elif act == "cos":
            self.act = torch.cos
        else:
            raise ValueError(f"Unknown activation function: {act}")

    @jit.export
    def forward(self, t: Tensor) -> Tensor:
        r""".. Signature:: ``... -> (..., d)``.

        Parameters
        ----------
        t: Tensor

        Returns
        -------
        Tensor
        """
        z = torch.einsum("..., k -> ...k", t, self.freq) + self.phase
        z = self.act(z)
        return torch.cat([t.unsqueeze(dim=-1), z], dim=-1)

    @jit.export
    def inverse(self, z: Tensor) -> Tensor:
        r""".. Signature:: ``(..., d) -> ...``.

        Parameters
        ----------
        z: Tensor

        Returns
        -------
        Tensor
        """
        return z[..., 0]


@autojit
class PositionalEncoder(nn.Module):
    r"""Positional encoding.

    .. math::
        x_{2 k}(t)   &:=\sin \left(\frac{t}{t^{2 k / τ}}\right) \\
        x_{2 k+1}(t) &:=\cos \left(\frac{t}{t^{2 k / τ}}\right)
    """

    HP: dict = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__doc__": __doc__,
        "__module__": __module__,  # type: ignore[name-defined]
        "num_dim": int,
        "scale": float,
    }

    # Constants
    num_dim: Final[int]
    r"""Number of dimensions."""
    scale: Final[float]
    r"""Scale factor for positional encoding."""

    # Buffers
    scales: Tensor
    r"""Scale factors for positional encoding."""

    def __init__(self, num_dim: int, *, scale: float) -> None:
        super().__init__()
        assert num_dim % 2 == 0, "num_dim must be even"
        self.num_dim = num_dim
        self.scale = float(scale)
        scales = self.scale ** (-2 * torch.arange(0, num_dim // 2) / (num_dim - 2))
        assert scales[0] == 1.0, "Something went wrong."
        self.register_buffer("scales", scales)

    @jit.export
    def forward(self, t: Tensor) -> Tensor:
        r""".. Signature:: ``... -> (..., 2d)``.

        Note: we simple concatenate the sin and cosine terms without interleaving them.

        Parameters
        ----------
        t: Tensor

        Returns
        -------
        Tensor
        """
        z = torch.einsum("..., d -> ...d", t, self.scales)
        return torch.cat([torch.sin(z), torch.cos(z)], dim=-1)

    @jit.export
    def inverse(self, t: Tensor) -> Tensor:
        r""".. Signature:: ``(..., 2d) -> ...``.

        Parameters
        ----------
        t: Tensor

        Returns
        -------
        Tensor
        """
        return torch.asin(t[..., 0])
