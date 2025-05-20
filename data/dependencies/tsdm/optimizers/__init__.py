r"""Utilities for optimizers."""

__all__ = [
    # Constants
    "Optimizer",
    "OPTIMIZERS",
    # Classes
    "LR_Scheduler",
    "LR_SCHEDULERS",
]

from typing import Final, TypeAlias

import torch.optim
from torch.optim import lr_scheduler

Optimizer: TypeAlias = torch.optim.Optimizer
r"""Type hint for optimizers."""

OPTIMIZERS: Final[dict[str, type[torch.optim.Optimizer]]] = {
    "Adadelta": torch.optim.Adadelta,
    # Implements Adadelta algorithm.
    "Adagrad": torch.optim.Adagrad,
    # Implements Adagrad algorithm.
    "Adam": torch.optim.Adam,
    # Implements Adam algorithm.
    "AdamW": torch.optim.AdamW,
    # Implements AdamW algorithm.
    "SparseAdam": torch.optim.SparseAdam,
    # Implements lazy version of Adam algorithm suitable for sparse tensors.
    "Adamax": torch.optim.Adamax,
    # Implements Adamax algorithm (a variant of Adam based on infinity norm).
    "ASGD": torch.optim.ASGD,
    # Implements Averaged Stochastic Gradient Descent.
    "LBFGS": torch.optim.LBFGS,
    # Implements L-BFGS algorithm, heavily inspired by minFunc.
    "RMSprop": torch.optim.RMSprop,
    # Implements RMSprop algorithm.
    "Rprop": torch.optim.Rprop,
    # Implements the resilient backpropagation algorithm.
    "SGD": torch.optim.SGD,
    # Implements stochastic gradient descent (optionally with momentum).
}
r"""Dictionary of all available optimizers."""

# noinspection PyProtectedMember
LR_Scheduler = lr_scheduler._LRScheduler  # pylint: disable=protected-access
r"""Type hint for lr_schedulers."""

LR_SCHEDULERS: Final[dict[str, type[lr_scheduler._LRScheduler]]] = {
    "LambdaLR": lr_scheduler.LambdaLR,
    "MultiplicativeLR": lr_scheduler.MultiplicativeLR,  # type: ignore[attr-defined]
    "StepLR": lr_scheduler.StepLR,
    "MultiStepLR": lr_scheduler.MultiStepLR,
    "ExponentialLR": lr_scheduler.ExponentialLR,
    "CosineAnnealingLR": lr_scheduler.CosineAnnealingLR,
    # "ReduceLROnPlateau": lr_scheduler.ReduceLROnPlateau,   # not subclass of _LRScheduler...
    "CyclicLR": lr_scheduler.CyclicLR,
    "OneCycleLR": lr_scheduler.OneCycleLR,  # type: ignore[attr-defined]
    "CosineAnnealingWarmRestarts": lr_scheduler.CosineAnnealingWarmRestarts,
}
r"""Dictionary of all available lr_schedulers."""
