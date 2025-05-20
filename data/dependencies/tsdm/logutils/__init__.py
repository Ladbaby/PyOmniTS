r"""Logging Utility Functions."""

__all__ = [
    # Constants
    "Logger",
    "LOGGERS",
    # Functions
    "compute_metrics",
    "log_optimizer_state",
    "log_kernel_information",
    "log_model_state",
    "log_metrics",
    "log_values",
    # Classes
    "StandardLogger",
]

from collections.abc import Callable
from typing import Final, TypeAlias

from data.dependencies.tsdm.logutils._logutils import (
    StandardLogger,
    compute_metrics,
    log_kernel_information,
    log_metrics,
    log_model_state,
    log_optimizer_state,
    log_values,
)

Logger: TypeAlias = Callable[..., None]

LOGGERS: Final[dict[str, Logger]] = {
    "log_optimizer_state": log_optimizer_state,
    "log_kernel_information": log_kernel_information,
    "log_model_state": log_model_state,
    "log_metrics": log_metrics,
}
