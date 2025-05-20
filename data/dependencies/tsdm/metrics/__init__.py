r"""Implementation of loss functions.

Theory
------
We define the following

1. A metric is a  function

    .. math:: 𝔪： 𝓟_0(𝓨×𝓨) ⟶ ℝ_{≥0}

    such that $𝔪(\{ (y_i, \hat{y}_i) ∣ i=1:n \}) = 0$ if and only if $y_i=\hat{y}_i∀i$

2. A metric is called decomposable, if it can be written as a function

    .. math
        𝔪 = Ψ∘(ℓ×𝗂𝖽)
        ℓ： 𝓨×𝓨 ⟶ ℝ_{≥0}
        Ψ： 𝓟_0(ℝ_{≥0}) ⟶ ℝ_{≥0}

    I.e. the function $ℓ$ is applied element-wise to all pairs $(y, \hat{y}$ and the function $Ψ$
    "accumulates" the results. Oftentimes, $Ψ$ is just the sum/mean/expectation value, although
    other accumulations such as the median value are also possible.

3. A metric is called instance-wise, if it can be written in the form

    .. math::
        𝔪： 𝓟_0(𝓨×𝓨) ⟶ ℝ_{≥ 0}, 𝔪(\{(y_i, \hat{y}_i) ∣  i=1:n \})
        = ∑_{i=1}^n ω(i, n)ℓ(y_i, \hat{y}_i)

4. A metric is called a loss-function, if and only if

   - It is differentiable almost everywhere.
   - It is non-constant, at least on some open set.

Note that in the context of time-series, we allow the accumulator to depend on the time variable.

Notes
-----
Contains losses in both modular and functional form.
  - See `tsdm.losses.functional` for functional implementations.
  - See `tsdm.losses` for modular implementations.
"""

__all__ = [
    # Sub-Modules
    "functional",
    # Types
    "Loss",
    "FunctionalLoss",
    "ModularLoss",
    # Constants
    "LOSSES",
    "FUNCTIONAL_LOSSES",
    "ModularLosses",
    # Classes
    "ND",
    "NRMSE",
    "Q_Quantile",
    "Q_Quantile_Loss",
    "WRMSE",
    "RMSE",
    # Functions
    "nd",
    "nrmse",
    "rmse",
    "q_quantile",
    "q_quantile_loss",
]


from typing import Final, TypeAlias

from torch import nn

from data.dependencies.tsdm.metrics._modular import ND, NRMSE, RMSE, WRMSE, Q_Quantile, Q_Quantile_Loss
from data.dependencies.tsdm.metrics.functional import (
    FUNCTIONAL_LOSSES,
    FunctionalLoss,
    nd,
    nrmse,
    q_quantile,
    q_quantile_loss,
    rmse,
)

ModularLoss: TypeAlias = nn.Module
r"""Type hint for modular losses."""

Loss: TypeAlias = FunctionalLoss | ModularLoss
r"""Type hint for losses."""

TORCH_LOSSES: Final[dict[str, type[nn.Module]]] = {
    "L1": nn.L1Loss,
    "CosineEmbedding": nn.CosineEmbeddingLoss,
    "CrossEntropy": nn.CrossEntropyLoss,
    "CTC": nn.CTCLoss,
    "NLL": nn.NLLLoss,
    "PoissonNLL": nn.PoissonNLLLoss,
    "GaussianNLL": nn.GaussianNLLLoss,
    "KLDiv": nn.KLDivLoss,
    "BCE": nn.BCELoss,
    "BCEWithLogits": nn.BCEWithLogitsLoss,
    "MarginRanking": nn.MarginRankingLoss,
    "MSE": nn.MSELoss,
    "HingeEmbedding": nn.HingeEmbeddingLoss,
    "Huber": nn.HuberLoss,
    "SmoothL1": nn.SmoothL1Loss,
    "SoftMargin": nn.SoftMarginLoss,
    "MultiMargin": nn.MultiMarginLoss,
    "MultiLabelMargin": nn.MultiLabelMarginLoss,
    "MultiLabelSoftMargin": nn.MultiLabelSoftMarginLoss,
    "TripletMargin": nn.TripletMarginLoss,
    "TripletMarginWithDistance": nn.TripletMarginWithDistanceLoss,
}
r"""Dictionary of all available modular losses in torch."""

TORCH_ALIASES: Final[dict[str, type[nn.Module]]] = {
    "MAE": nn.L1Loss,
    "L2": nn.MSELoss,
    "XENT": nn.CrossEntropyLoss,
    "KL": nn.KLDivLoss,
}
r"""Dictionary containing additional aliases for modular losses in torch."""

ModularLosses: Final[dict[str, type[nn.Module]]] = {
    "ND": ND,
    "NRMSE": NRMSE,
    "Q_Quantile": Q_Quantile,
    "Q_Quantile_Loss": Q_Quantile_Loss,
    "RMSE": RMSE,
} | (TORCH_LOSSES | TORCH_ALIASES)
r"""Dictionary of all available modular losses."""


LOSSES: Final[dict[str, FunctionalLoss | type[ModularLoss]]] = {
    **FUNCTIONAL_LOSSES,
    **ModularLosses,
}
r"""Dictionary of all available losses."""
