r"""Utility functions.

TODO:  Module description
"""

# from __future__ import annotations


__all__ = [
    # Constants
    # Classes
    "Split",
    # Functions
    "deep_dict_update",
    "deep_kval_update",
    "flatten_dict",
    "flatten_nested",
    "initialize_from",
    "initialize_from_config",
    "is_partition",
    "now",
    "paths_exists",
    "prepend_path",
    "round_relative",
    "pairwise_disjoint",
    "pairwise_disjoint_masks",
]

import os
from collections.abc import Callable, Collection, Hashable, Iterable, Mapping, Sequence
from datetime import datetime
from functools import partial
from importlib import import_module
from logging import getLogger
from pathlib import Path
from typing import Any, Literal, NamedTuple, Optional, Union, overload

import numpy as np
from numpy.typing import NDArray
from torch import nn

from data.dependencies.tsdm.utils.types import AnyTypeVar, Nested, ObjectVar, PathType, ReturnVar
from data.dependencies.tsdm.utils.types.abc import HashableType

__logger__ = getLogger(__name__)

EMPTY_PATH: Path = Path()
r"""Constant: Blank path."""


def pairwise_disjoint(sets: Iterable[set]) -> bool:
    r"""Check if sets are pairwise disjoint."""
    union = set().union(*sets)
    return len(union) == sum(len(s) for s in sets)


def pairwise_disjoint_masks(masks: Iterable[NDArray[np.bool_]]) -> bool:
    r"""Check if masks are pairwise disjoint."""
    return all(sum(masks) == 1)  # type: ignore[arg-type]


def flatten_dict(
    d: dict[Any, Any],
    /,
    *,
    join_string: Optional[str] = None,
    key_func: Optional[Callable[[Hashable, Hashable], Hashable]] = None,
    recursive: bool | int = True,
) -> dict[Any, Any]:
    r"""Flatten a dictionary containing iterables to a list of tuples.

    Parameters
    ----------
    d: Input Dictionary
    join_string:
        use this option in case of nested dicts with string keys,
    key_func:
        function that creates new key from old key and subkey
    recursive:
        If true applies flattening strategy recursively on nested dicts.
        If int, specifies the maximum depth of recursion.
    """
    if join_string is not None and key_func is not None:
        raise ValueError("Only one of join_string or key_func can be specified.")

    if join_string is not None:
        key_func = lambda k, sk: f"{k}{join_string}{sk}"  # noqa: E731
    elif key_func is None:
        key_func = lambda k, sk: (k, sk)  # noqa: E731

    result = {}
    for key, item in d.items():
        if isinstance(key, tuple):
            raise ValueError("Keys are not allowed to be tuples!")
        if isinstance(item, dict) and recursive:
            subdict = flatten_dict(
                item, recursive=True, key_func=key_func, join_string=join_string
            )
            for subkey, subitem in subdict.items():
                result[key_func(key, subkey)] = subitem
        else:
            result[key] = item
    return result


class Split(NamedTuple):
    r"""Holds indices for train/valid/test set."""

    train: int
    """The training set."""
    valid: Any
    """The validation set."""
    test: Any
    """The testing set."""


def round_relative(x: np.ndarray, decimals: int = 2) -> np.ndarray:
    r"""Round to relative precision."""
    order = np.where(x == 0, 0, np.floor(np.log10(x)))
    digits = decimals - order
    rounded = np.rint(x * 10**digits)
    return np.true_divide(rounded, 10**digits)


def now():
    r"""Return current time in iso format."""
    return datetime.now().isoformat(timespec="seconds")


def deep_dict_update(d: dict, new_kvals: Mapping) -> dict:
    r"""Update nested dictionary recursively in-place with new dictionary.

    Reference: https://stackoverflow.com/a/30655448/9318372

    Parameters
    ----------
    d: dict
    new_kvals: Mapping
    """
    # if not inplace:
    #     return deep_dict_update(deepcopy(d), new_kvals, inplace=False)

    for key, value in new_kvals.items():
        if isinstance(value, Mapping) and value:
            d[key] = deep_dict_update(d.get(key, {}), value)
        else:
            # if value is not None or not safe:
            d[key] = new_kvals[key]
    return d


def deep_kval_update(d: dict, **new_kvals: dict) -> dict:
    r"""Update nested dictionary recursively in-place with key-value pairs.

    Reference: https://stackoverflow.com/a/30655448/9318372

    Parameters
    ----------
    d: dict
    new_kvals: dict
    """
    # if not inplace:
    #     return deep_dict_update(deepcopy(d), new_kvals, inplace=False)

    for key, value in d.items():
        if isinstance(value, Mapping) and value:
            d[key] = deep_kval_update(d.get(key, {}), **new_kvals)
        elif key in new_kvals:
            # if value is not None or not safe:
            d[key] = new_kvals[key]
    return d


def apply_nested(
    nested: Nested[AnyTypeVar | None],
    kind: type[AnyTypeVar],
    func: Callable[[AnyTypeVar], ReturnVar],
) -> Nested[ReturnVar | None]:
    r"""Apply function to nested iterables of a given kind.

    Parameters
    ----------
    nested: Nested Data-Structure (Iterable, Mapping, ...)
    kind: The type of the leave nodes
    func: A function to apply to all leave Nodes
    """
    if nested is None:
        return None
    if isinstance(nested, kind):
        return func(nested)
    if isinstance(nested, Mapping):
        return {k: apply_nested(v, kind, func) for k, v in nested.items()}
    # TODO https://github.com/python/mypy/issues/11615
    if isinstance(nested, Collection):
        return [apply_nested(obj, kind, func) for obj in nested]  # type: ignore[misc]
    raise TypeError(f"Unsupported type: {type(nested)}")


@overload
def prepend_path(
    files: Nested[PathType],
    parent: Path,
    *,
    keep_none: bool = False,
) -> Nested[Path]:
    ...


@overload
def prepend_path(
    files: Nested[Optional[PathType]],
    parent: Path,
    *,
    keep_none: Literal[False] = False,
) -> Nested[Path]:
    ...


@overload
def prepend_path(
    files: Nested[Optional[PathType]],
    parent: Path,
    *,
    keep_none: Literal[True] = True,
) -> Nested[Optional[Path]]:
    ...


def prepend_path(
    files: Nested[Optional[PathType]],
    parent: Path,
    *,
    keep_none: bool = False,
) -> Nested[Optional[Path]]:
    r"""Prepends path to all files in nested iterable.

    Parameters
    ----------
    files
        Nested datastructures with Path-objects at leave nodes.
    parent:
    keep_none:
        If True, None-values are kept.

    Returns
    -------
    Nested data-structure with prepended path.
    """
    # TODO: change it to apply_nested in python 3.10

    if files is None:
        return None if keep_none else parent
    if isinstance(files, (str, Path, os.PathLike)):
        return parent / Path(files)
    if isinstance(files, Mapping):
        return {
            k: prepend_path(v, parent, keep_none=keep_none) for k, v in files.items()  # type: ignore[arg-type]
        }
    # TODO https://github.com/python/mypy/issues/11615
    if isinstance(files, Collection):
        return [prepend_path(f, parent, keep_none=keep_none) for f in files]  # type: ignore[misc,arg-type]
    raise TypeError(f"Unsupported type: {type(files)}")


def flatten_nested(nested: Any, kind: type[HashableType]) -> set[HashableType]:
    r"""Flatten nested iterables of a given kind.

    Parameters
    ----------
    nested: Any
    kind: hashable

    Returns
    -------
    set[hashable]
    """
    if nested is None:
        return set()
    if isinstance(nested, kind):
        return {nested}
    if isinstance(nested, Mapping):
        return set.union(*(flatten_nested(v, kind) for v in nested.values()))
    if isinstance(nested, Iterable):
        return set.union(*(flatten_nested(v, kind) for v in nested))
    raise ValueError(f"{type(nested)} is not understood")


def initialize_from_config(config: dict[str, Any]) -> nn.Module:
    r"""Initialize `nn.Module` from config object."""
    assert "__name__" in config, "__name__ not found in dict"
    assert "__module__" in config, "__module__ not found in dict"
    __logger__.debug("Initializing %s", config)
    config = config.copy()
    module = import_module(config.pop("__module__"))
    cls = getattr(module, config.pop("__name__"))
    opts = {key: val for key, val in config.items() if not is_dunder("key")}
    return cls(**opts)


# partial from type
@overload
def initialize_from(  # type: ignore[misc]
    lookup_table: dict[str, type[ObjectVar]],
    /,
    __name__: str,
    **kwargs: Any,
) -> ObjectVar:
    ...


# partial from func
@overload
def initialize_from(
    lookup_table: dict[str, Callable[..., ReturnVar]],
    /,
    __name__: str,
    **kwargs: Any,
) -> Callable[..., ReturnVar]:
    ...


# partial from type
# @overload
# def initialize_from(
#     lookup_table: dict[str, Union[type[ObjectType], Callable[..., ReturnType]]],
#     /,
#     __name__: str,
#     **kwargs: Any,
# ) -> Union[ObjectType, Callable[..., ReturnType]]:
#     ...


def initialize_from(  # type: ignore[misc]
    lookup_table: Union[
        dict[str, type[ObjectVar]],
        dict[str, Callable[..., ReturnVar]],
        dict[str, type[ObjectVar] | Callable[..., ReturnVar]],
    ],
    /,
    __name__: str,
    **kwargs: Any,
) -> ObjectVar | Callable[..., ReturnVar]:
    r"""Lookup class/function from dictionary and initialize it.

    Roughly equivalent to:

    .. code-block:: python

        obj = lookup_table[__name__]
        if isclass(obj):
            return obj(**kwargs)
        return partial(obj, **kwargs)

    Parameters
    ----------
    lookup_table: dict[str, Callable]
    __name__: str
        The name of the class/function
    kwargs: Any
        Optional arguments to initialize class/function

    Returns
    -------
    object
        The initialized class/function
    """
    obj = lookup_table[__name__]
    assert callable(obj), f"Looked up object {obj} not callable class/function."

    # check that obj is a class, but not metaclass or instance.
    if isinstance(obj, type) and not issubclass(obj, type):
        initialized_object: ObjectVar = obj(**kwargs)
        return initialized_object
    # if it is function, fix kwargs
    initialized_callable: Callable[..., ReturnVar] = partial(obj, **kwargs)  # type: ignore[assignment]
    return initialized_callable


def is_dunder(name: str) -> bool:
    r"""Check if name is a dunder method.

    Parameters
    ----------
    name: str

    Returns
    -------
    bool
    """
    return name.isidentifier() and name.startswith("__") and name.endswith("__")


def is_partition(*partition: Collection, union: Optional[Sequence] = None) -> bool:
    r"""Check if partition is a valid partition of union."""
    if len(partition) == 1:
        return is_partition(*next(iter(partition)), union=union)

    sets = (set(p) for p in partition)
    part_union = set().union(*sets)

    if union is not None and part_union != set(union):
        return False
    return len(part_union) == sum(len(p) for p in partition)


def initialize_module_from_config(config: dict[str, Any]) -> nn.Module:
    r"""Initialize a class from a dictionary."""
    assert "__name__" in config, "__name__ not found in dict"
    assert "__module__" in config, "__module__ not found in dict"
    __logger__.debug("Initializing %s", config)
    config = config.copy()
    module = import_module(config.pop("__module__"))
    cls = getattr(module, config.pop("__name__"))
    opts = {key: val for key, val in config.items() if not is_dunder("key")}
    return cls(**opts)


def paths_exists(
    paths: Nested[Optional[PathType]],
    *,
    parent: Path = EMPTY_PATH,
) -> bool:
    r"""Check whether the files exist.

    The input can be arbitrarily nested data-structure with `Path` in leaves.

    Parameters
    ----------
    paths: None | Path | Collection[Path] | Mapping[Any, None | Path | Collection[Path]]
    parent: Path = Path(),

    Returns
    -------
    bool
    """
    if isinstance(paths, str):
        return Path(paths).exists()
    if paths is None:
        return True
    if isinstance(paths, Mapping):
        return all(paths_exists(f, parent=parent) for f in paths.values())
    if isinstance(paths, Collection):
        return all(paths_exists(f, parent=parent) for f in paths)
    if isinstance(paths, Path):
        return (parent / paths).exists()

    raise ValueError(f"Unknown type for rawdata_file: {type(paths)}")
