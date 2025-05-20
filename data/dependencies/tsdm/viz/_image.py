r"""Visualization Utilities for image data."""

__all__ = [
    # Functions
    "kernel_heatmap",
]

from typing import Literal

import numpy as np
import torch
from matplotlib import cm
from numpy.typing import NDArray
from torch import Tensor


@torch.no_grad()
def kernel_heatmap(
    kernel: NDArray | Tensor,
    fmt: Literal["HWC", "CHW"] = "HWC",
    cmap: str = "seismic",
) -> NDArray:
    r"""Create heatmap of given matrix.

    .. Signature:: ``(..., ) ⟶ (..., 3)`` if "HWC" or ``(..., ) ⟶ (3, ...)`` if "CHW".

    By default, the data is linearly transformed to a normal distribution $𝓝(½,⅙)$,
    which ensures that 99.7% of the data lies in the interval $[0,1]$, and then clipped.

    Parameters
    ----------
    kernel: NDArray or Tensor
    fmt: Literal["HWC", "CHW"] = "HWC"
        Whether to put channels first or last.
    cmap: str = "seismic"
        The colormap.

    Returns
    -------
    NDArray
    """
    # This transformation is chosen because by the 68–95–99.7 rule,
    # for k=6=2⋅3 roughly 99.7% of the probability mass will lie in the interval [0, 1]
    kernel = 0.5 + (kernel - kernel.mean()) / (6 * kernel.std())
    kernel = kernel.clip(0, 1)

    if isinstance(kernel, Tensor):
        kernel = kernel.cpu().numpy()

    colormap = cm.get_cmap(cmap)
    RGBA = colormap(kernel)
    RGB = RGBA[..., :-1]

    if fmt == "HWC":
        return RGB
    if fmt == "CHW":
        return np.rollaxis(RGB, -1)
    raise ValueError(fmt)
