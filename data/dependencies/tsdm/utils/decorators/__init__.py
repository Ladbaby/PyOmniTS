r"""Submodule containing general purpose decorators.

#TODO add module description.
"""

__all__ = [
    # Classes
    # Constants
    "Decorator",
    "DECORATORS",
    # Functions
    "autojit",
    "decorator",
    # "sphinx_value",
    "timefun",
    "trace",
    "vectorize",
    "IterItems",
    "IterKeys",
    "wrap_func",
]

from collections.abc import Callable
from typing import Final

from data.dependencies.tsdm.utils.decorators._decorators import (
    IterItems,
    IterKeys,
    autojit,
    decorator,
    timefun,
    trace,
    vectorize,
    wrap_func,
)

Decorator = Callable[..., Callable]
r"""Type hint for dataset."""

DECORATORS: Final[dict[str, Decorator]] = {
    "autojit": autojit,
    "decorator": decorator,
    # "sphinx_value": sphinx_value,
    "timefun": timefun,
    "trace": trace,
}
r"""Dictionary of all available decorators."""
