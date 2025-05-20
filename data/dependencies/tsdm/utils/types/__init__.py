r"""Generic types for type hints etc."""

__all__ = [
    # Submodules
    "abc",
    "protocols",
    "time",
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


from data.dependencies.tsdm.utils.types import abc, protocols, time
from data.dependencies.tsdm.utils.types._types import (
    AliasVar,
    AnyTypeVar,
    Args,
    ClassVar,
    DtypeVar,
    KeyVar,
    KWArgs,
    ModuleVar,
    Nested,
    ObjectVar,
    PandasObject,
    PandasVar,
    Parameters,
    PathType,
    PathVar,
    ReturnVar,
    Self,
    TensorVar,
    TorchModuleVar,
    ValueVar,
)
