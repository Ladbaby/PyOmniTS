r"""Provides utility functions."""

__all__ = [
    # Sub-Packages
    "data",
    "decorators",
    "types",
    # Sub-Modules
    "remote",
    "system",
    # Constants
    # Classes
    "PatchedABCMeta",
    "Split",
    "LazyDict",
    "LazyFunction",
    # decorators
    "abstractattribute",
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
]

from data.dependencies.tsdm.utils import data, decorators, remote, system, types
from data.dependencies.tsdm.utils._subclassing import PatchedABCMeta, abstractattribute
from data.dependencies.tsdm.utils._util import (
    Split,
    deep_dict_update,
    deep_kval_update,
    flatten_dict,
    flatten_nested,
    initialize_from,
    initialize_from_config,
    is_partition,
    now,
    pairwise_disjoint,
    paths_exists,
    prepend_path,
    round_relative,
)
from data.dependencies.tsdm.utils.lazydict import LazyDict, LazyFunction
