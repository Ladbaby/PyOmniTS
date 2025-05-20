r"""Generic types for type hints etc."""

# from __future__ import annotations

__all__ = [
    # Type Variables
    "AnyTypeVar",
    "AliasVar",
    "ClassVar",
    "DtypeVar",
    "KeyVar",
    "ModuleVar",
    "ObjectVar",
    "PandasVar",
    "PathVar",
    "ReturnVar",
    "Self",
    "TensorVar",
    "TorchModuleVar",
    "ValueVar",
    # Type Aliases
    "Args",
    "KWArgs",
    "Nested",
    "PandasObject",
    "PathType",
    # ParamSpec
    "Parameters",
]

import os
from collections.abc import Collection, Hashable, Mapping
from pathlib import Path
from types import GenericAlias, ModuleType
from typing import Any, ParamSpec, TypeAlias, TypeVar

from numpy import ndarray
from pandas import DataFrame, Index, Series
from torch import Tensor, nn

Parameters = ParamSpec("Parameters")
r"""TypeVar for decorated function inputs values."""

# region TypeVars

AnyTypeVar = TypeVar("AnyTypeVar")
r"""Type Variable arbitrary types.."""

AliasVar = TypeVar("AliasVar", bound=GenericAlias)
r"""Type Variable `TypeAlias`."""

ClassVar = TypeVar("ClassVar", bound=type)
r"""Type Variable for `type`."""

DtypeVar = TypeVar("DtypeVar")
r"""Type Variable for `Dtype`."""

KeyVar = TypeVar("KeyVar", bound=Hashable)
r"""Type Variable for `Mapping` keys."""

ModuleVar = TypeVar("ModuleVar", bound=ModuleType)
r"""Type Variable for Modules."""

ObjectVar = TypeVar("ObjectVar", bound=object)
r"""Type Variable for `object`."""

PandasVar = TypeVar("PandasVar", Index, Series, DataFrame)
r"""Type Variable for `pandas` objects."""

PathVar = TypeVar("PathVar", str, Path, os.PathLike[str])
r"""Type Variable for path-like objects."""

ReturnVar = TypeVar("ReturnVar")
r"""Type Variable for return values."""

Self = TypeVar("Self")
r"""Type Variable for for self reference."""  # FIXME: PEP673 @ python3.11

TensorVar = TypeVar("TensorVar", Tensor, ndarray)
r"""Type Variable for `torch.Tensor` or `numpy.ndarray` objects."""

TorchModuleVar = TypeVar("TorchModuleVar", bound=nn.Module)
r"""Type Variable for `torch.nn.Modules`."""

ValueVar = TypeVar("ValueVar")
r"""Type Variable for `Mapping` values."""

# endregion

# region TypeAliases

# FIXME: https://github.com/python/mypy/pull/13297 Recursive Alias
# NestedMapping: TypeAlias = Mapping[KeyVar, ValueVar | Mapping[KeyVar, 'NestedMapping']]
# r"""Generic Type Alias for nested `Mapping`."""
#
# NestedDict: TypeAlias = dict[KeyVar, ValueVar | dict[KeyVar, 'NestedDict']]
# r"""Generic Type Alias for nested `dict`."""
#
# NestedTuple: TypeAlias = tuple[ValueVar | tuple['NestedTuple', ...], ...]
# r"""Generic Type Alias for nested `tuple`."""
#
# NestedList: TypeAlias = list[ValueVar | list['NestedList']]
# r"""GenericType Alias for nested `list`."""
#
# NestedSet: TypeAlias = set[ValueVar | set['NestedSet']]
# r"""Generic Type Alias for nested `set`."""
#
# nested: TypeAlias = AliasVar[ValueVar | AliasVar[ValueVar]]
# r"""Generic Type Alias for generic nested structure."""


Args: TypeAlias = tuple[AnyTypeVar, ...]
r"""Type Alias for `*args`."""

KWArgs: TypeAlias = dict[str, AnyTypeVar]
r"""Type Alias for `**kwargs`."""

Nested: TypeAlias = AnyTypeVar | Collection[AnyTypeVar] | Mapping[Any, AnyTypeVar]
r"""Type Alias for nested types (JSON-Like)."""

PandasObject: TypeAlias = DataFrame | Series | Index
r"""Type Alias for `pandas` objects."""

PathType: TypeAlias = str | Path  # | os.PathLike[str]
r"""Type Alias for path-like objects."""

# endregion
