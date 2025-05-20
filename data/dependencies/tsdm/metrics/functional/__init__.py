r"""Implementations of loss functions.

Notes
-----
Contains losses in functional form.
"""

__all__ = [
    # Types
    "FunctionalLoss",
    # Constants
    "FUNCTIONAL_LOSSES",
    # Functions
    "nd",
    "nrmse",
    "rmse",
    "q_quantile",
    "q_quantile_loss",
]

from collections.abc import Callable
from typing import Final, TypeAlias

from torch import Tensor, nn

from data.dependencies.tsdm.metrics.functional._functional import (
    nd,
    nrmse,
    q_quantile,
    q_quantile_loss,
    rmse,
)

# TODO: use better definition [Tensor, Tensor, ...] -> Tensor once supported
FunctionalLoss: TypeAlias = Callable[..., Tensor]
r"""Type hint for functional losses."""

TORCH_FUNCTIONAL_LOSSES: Final[dict[str, FunctionalLoss]] = {
    "binary_cross_entropy": nn.functional.binary_cross_entropy,
    # Function that measures the Binary Cross Entropy between the target and the output.
    "binary_cross_entropy_with_logits": nn.functional.binary_cross_entropy_with_logits,
    # Function that measures Binary Cross Entropy between target and output logits.
    "poisson_nll": nn.functional.poisson_nll_loss,
    # Poisson negative log likelihood loss.
    "cosine_embedding": nn.functional.cosine_embedding_loss,
    # See CosineEmbeddingLoss for details.
    "cross_entropy": nn.functional.cross_entropy,
    # This criterion combines log_softmax and nll_loss in a single function.
    "ctc_loss": nn.functional.ctc_loss,
    # The Connectionist Temporal Classification loss.
    "gaussian_nll": nn.functional.gaussian_nll_loss,
    # Gaussian negative log likelihood loss.
    "hinge_embedding": nn.functional.hinge_embedding_loss,
    # See HingeEmbeddingLoss for details.
    "kl_div": nn.functional.kl_div,
    # The Kullback-Leibler divergence Loss
    "l1": nn.functional.l1_loss,
    # Function that takes the mean element-wise absolute value difference.
    "mse": nn.functional.mse_loss,
    # Measures the element-wise mean squared error.
    "margin_ranking": nn.functional.margin_ranking_loss,
    # See MarginRankingLoss for details.
    "multilabel_margin": nn.functional.multilabel_margin_loss,
    # See MultiLabelMarginLoss for details.
    "multilabel_soft_margin": nn.functional.multilabel_soft_margin_loss,
    # See MultiLabelSoftMarginLoss for details.
    "multi_margin": nn.functional.multi_margin_loss,
    # multi_margin_loss(input, target, p=1, margin=1, weight=None, size_average=None,
    "nll": nn.functional.nll_loss,
    # The negative log likelihood loss.
    "huber": nn.functional.huber_loss,
    # Function that uses a squared term if the absolute element-wise error falls below
    # delta and a delta-scaled L1 term otherwise.
    "smooth_l1": nn.functional.smooth_l1_loss,
    # Function that uses a squared term if the absolute element-wise error falls below
    # beta and an L1 term otherwise.
    "soft_margin": nn.functional.soft_margin_loss,
    # See SoftMarginLoss for details.
    "triplet_margin": nn.functional.triplet_margin_loss,
    # See TripletMarginLoss for details
    "triplet_margin_with_distance": nn.functional.triplet_margin_with_distance_loss,
    # See TripletMarginWithDistanceLoss for details.
}
r"""Dictionary of all available losses in torch."""

TORCH_ALIASES: Final[dict[str, FunctionalLoss]] = {
    "mae": nn.functional.l1_loss,
    "l2": nn.functional.mse_loss,
    "xent": nn.functional.cross_entropy,
    "kl": nn.functional.kl_div,
}
r"""Dictionary containing additional aliases for losses in torch."""

FUNCTIONAL_LOSSES: Final[dict[str, FunctionalLoss]] = {
    "nd": nd,
    "nrmse": nrmse,
    "q_quantile": q_quantile,
    "q_quantile_loss": q_quantile_loss,
} | (TORCH_FUNCTIONAL_LOSSES | TORCH_ALIASES)
r"""Dictionary of all available functional losses."""
