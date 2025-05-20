r"""Implementations of loss functions.

Notes
-----
Contains losses in modular form.
  - See `tsdm.losses.functional` for functional implementations.
"""

__all__ = [
    # Classes
    "ND",
    "NRMSE",
    "Q_Quantile",
    "Q_Quantile_Loss",
    "WRMSE",
    "RMSE",
]


from typing import Final

import numpy as np
import torch
from torch import Tensor, jit, nn

from data.dependencies.tsdm.metrics.functional import nd, nrmse, q_quantile, q_quantile_loss
from data.dependencies.tsdm.utils.decorators import autojit


@autojit
class ND(nn.Module):
    r"""Compute the normalized deviation score.

    .. math:: 𝖭𝖣(x, x̂) = \frac{∑_{t,k} |x̂_{t,k} -  x_{t,k}|}{∑_{t,k} |x_{t,k}|}

    TODO: How to distinguish batch univariate vs single multivariate?
    => Batch makes little sense since all could have different length!

    References
    ----------
    - | Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction
      | https://papers.nips.cc/paper/2016/hash/85422afb467e9456013a2a51d4dff702-Abstract.html
    - | N-BEATS: Neural basis expansion analysis for interpretable time series forecasting
      | https://openreview.net/forum?id=r1ecqn4YwB
    """

    @jit.export
    def forward(self, x: Tensor, xhat: Tensor) -> Tensor:
        r"""Compute the loss value.

        Parameters
        ----------
        x: Tensor
        xhat: Tensor

        Returns
        -------
        Tensor
        """
        return nd(x, xhat)


@autojit
class NRMSE(nn.Module):
    r"""Compute the normalized deviation score.

    .. math:: 𝖭𝖱𝖬𝖲𝖤(x, x̂) = \frac{\sqrt{ \frac{1}{T}∑_{t,k} |x̂_{t,k} - x_{t,k}|^2 }}{∑_{t,k} |x_{t,k}|}

    References
    ----------
    - | Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction
      | https://papers.nips.cc/paper/2016/hash/85422afb467e9456013a2a51d4dff702-Abstract.html
    """

    @jit.export
    def forward(self, x: Tensor, xhat: Tensor) -> Tensor:
        r"""Compute the loss value.

        Parameters
        ----------
        x: Tensor
        xhat: Tensor

        Returns
        -------
        Tensor
        """
        return nrmse(x, xhat)


@autojit
class Q_Quantile(nn.Module):
    r"""The q-quantile.

    .. math:: 𝖯_q(x,x̂) = \begin{cases} q |x-x̂|:& x≥x̂ \\ (1-q)|x-x̂|:& x≤x̂ \end{cases}

    References
    ----------
    - | Deep State Space Models for Time Series Forecasting
      | https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html
    """

    @jit.export
    def forward(self, x: Tensor, xhat: Tensor) -> Tensor:
        r"""Compute the loss value.

        Parameters
        ----------
        x: Tensor
        xhat: Tensor

        Returns
        -------
        Tensor
        """
        return q_quantile(x, xhat)


@autojit
class Q_Quantile_Loss(nn.Module):
    r"""The q-quantile loss.

    .. math:: 𝖰𝖫_q(x,x̂) = 2\frac{∑_{it}𝖯_q(x_{it},x̂_{it})}{∑_{it}|x_{it}|}

    References
    ----------
    - | Deep State Space Models for Time Series Forecasting
      | https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html
    """

    @jit.export
    def forward(self, x: Tensor, xhat: Tensor) -> Tensor:
        r"""Compute the loss value.

        Parameters
        ----------
        x: Tensor
        xhat: Tensor

        Returns
        -------
        Tensor
        """
        return q_quantile_loss(x, xhat)


@autojit
class WRMSE(nn.Module):
    r"""Weighted Root Mean Square Error.

    .. math:: (1/m)∑_m (1/n)∑_n w(x_{n,m}- x_{n,m})^2
    """

    # Constants
    rank: Final[int]
    r"""CONST: The number of dimensions of the weight tensor."""
    shape: Final[tuple[int, ...]]
    r"""CONST: The shape of the weight tensor."""

    # Buffers
    w: Tensor
    r"""BUFFER: The weight-vector."""

    def __init__(
        self,
        w: Tensor,
        /,
        normalize: bool = True,
    ):
        r"""Compute the weighted RMSE.

        Channel-wise: RMSE(RMSE(channel))
        Non-channel-wise: RMSE(flatten(results))

        Parameters
        ----------
        w: Tensor
        normalize: bool = True
        """
        super().__init__()
        assert torch.all(w >= 0) and torch.any(w > 0)
        self.register_buffer("w", w / torch.sum(w) if normalize else w)
        self.w = self.w.to(dtype=torch.float32)
        self.rank = len(w.shape)
        self.register_buffer("FAILED", torch.tensor(float("nan")))
        self.shape = tuple(w.shape)

    @jit.export
    def forward(self, x: Tensor, xhat: Tensor) -> Tensor:
        r""".. Signature:: ``[(..., m), (..., m)] → ...``.

        Parameters
        ----------
        x: Tensor
        xhat: Tensor

        Returns
        -------
        Tensor
        """
        self.w = self.w.to(device=x.device)
        assert x.shape[-self.rank :] == self.shape
        # the residuals, shape: ...𝐦
        r = self.w * (x - xhat) ** 2

        return torch.sqrt(torch.mean(r))

    def __repr__(self) -> str:
        r"""Pretty print."""
        with np.printoptions(precision=2):
            weights = self.w.cpu().numpy()
            return f"{self.__class__.__name__}(\n" + repr(weights) + "\n)"


@autojit
class RMSE(nn.Module):
    r"""Root Mean Square Error."""

    mask_nan_targets: Final[bool]
    r"""CONST: Whether to mask NaN targets, not counting them as observations."""

    def __init__(self, mask_nan_targets: bool = True):
        r"""Compute the RMSE.

        Parameters
        ----------
        mask_nan_targets: bool = True
        """
        super().__init__()
        self.mask_nan_targets = mask_nan_targets

    @jit.export
    def forward(
        self,
        x: Tensor,
        xhat: Tensor,
    ) -> Tensor:
        r"""Compute the RMSE.

        .. math:: 𝗋𝗆𝗌𝖾(x,x̂) = \sqrt{𝔼[‖x - x̂‖^2]}
        """
        if self.mask_nan_targets:
            mask = torch.isnan(x)
            x = x[mask]
            xhat = xhat[mask]

        return torch.sqrt(torch.mean((x - xhat) ** 2))


@autojit
class MSE(nn.Module):
    r"""Root Mean Square Error."""

    mask_nan_targets: Final[bool]
    r"""CONST: Whether to mask NaN targets, not counting them as observations."""

    def __init__(self, mask_nan_targets: bool = True):
        r"""Compute the RMSE.

        Parameters
        ----------
        mask_nan_targets: bool = True
        """
        super().__init__()
        self.mask_nan_targets = mask_nan_targets

    @jit.export
    def forward(
        self,
        x: Tensor,
        xhat: Tensor,
    ) -> Tensor:
        r"""Compute the RMSE.

        .. math:: 𝗋𝗆𝗌𝖾(x,x̂) = \sqrt{𝔼[‖x - x̂‖^2]}
        """
        if self.mask_nan_targets:
            mask = torch.isnan(x)
            x = x[mask]
            xhat = xhat[mask]

        return torch.mean((x - xhat) ** 2)
