r"""Dtype data for numpy/pandas/torch."""

__all__ = [
    # TypeVars and TypeAliases
    "ScalarDType",
    # DTYPES
    "NUMPY_DTYPES",
    "TORCH_DTYPES",
    "PANDAS_DTYPES",
    "PYTHON_DTYPES",
    # TYPESTRINGS
    "NUMPY_TYPESTRINGS",
    "NUMPY_TYPECODES",
    "TORCH_TYPESTRINGS",
    "PANDAS_TYPESTRINGS",
    "PYTHON_TYPESTRINGS",
    # NUMPY TYPECODES
    "NUMPY_BOOL_TYPECODES",
    "NUMPY_COMPLEX_TYPECODES",
    "NUMPY_FLOAT_TYPECODES",
    "NUMPY_INT_TYPECODES",
    "NUMPY_OBJECT_TYPECODES",
    "NUMPY_STRING_TYPECODES",
    "NUMPY_TIME_TYPECODES",
    "NUMPY_UINT_TYPECODES",
    # NUMPY TYPESTRINGS
    "NUMPY_BOOL_TYPESTRINGS",
    "NUMPY_COMPLEX_TYPESTRINGS",
    "NUMPY_FLOAT_TYPESTRINGS",
    "NUMPY_INT_TYPESTRINGS",
    "NUMPY_OBJECT_TYPESTRINGS",
    "NUMPY_STRING_TYPESTRINGS",
    "NUMPY_TIME_TYPESTRINGS",
    "NUMPY_UINT_TYPESTRINGS",
    # TORCH TYPESTRINGS
    "TORCH_BOOL_TYPESTRINGS",
    "TORCH_COMPLEX_TYPESTRINGS",
    "TORCH_FLOAT_TYPESTRINGS",
    "TORCH_INT_TYPESTRINGS",
    "TORCH_UINT_TYPESTRINGS",
    # Constants
    "BOOLS",
    "CATEGORIES",
    "PRECISION",
    "EMOJIS",
    "STRINGS",
    "TYPESTRINGS",
]

from collections import namedtuple
from datetime import datetime, timedelta
from typing import Final, TypeAlias

import numpy as np
import pandas
import torch
from pandas.api.extensions import ExtensionDtype

# region numpy typecodes -----------------------------------------------------------------------------------------------

NUMPY_INT_TYPECODES: Final[dict[type[np.signedinteger], str]] = {
    np.int8: "b",
    np.int16: "h",
    np.int32: "i",
    np.int64: "l",
    np.byte: "b",
    np.short: "h",
    np.intc: "i",
    np.int_: "l",
    np.intp: "l",
    np.longlong: "q",
}
r"""Dictionary of all signed `numpy` integer data type typecodes."""

NUMPY_UINT_TYPECODES: Final[dict[type[np.unsignedinteger], str]] = {
    np.uint8: "B",
    np.uint16: "H",
    np.uint32: "I",
    np.uint64: "L",
    np.ubyte: "B",
    np.ushort: "H",
    np.uintc: "I",
    np.uint: "L",
    np.uintp: "L",
    np.ulonglong: "Q",
}
r"""Dictionary of all unsigned `numpy` integer data type typecodes."""

NUMPY_FLOAT_TYPECODES: Final[dict[type[np.floating], str]] = {
    np.float16: "e",
    np.float32: "f",
    np.float64: "d",
    np.float128: "g",
    np.half: "e",
    np.single: "f",
    np.double: "d",
    np.longdouble: "g",
}
r"""Dictionary of all `numpy` float data type typecodes."""

NUMPY_COMPLEX_TYPECODES: Final[dict[type[np.complexfloating], str]] = {
    np.complex64: "F",
    np.complex128: "D",
    np.complex256: "G",
    np.csingle: "F",
    np.cdouble: "D",
    np.clongdouble: "G",
}
r"""Dictionary of all `numpy` complex data types."""


NUMPY_TIME_TYPECODES: Final[dict[type[np.generic], str]] = {
    np.timedelta64: "M",  # timedelta64
    np.datetime64: "m",  # datetime64
}
r"""Dictionary of all `numpy` time data type typecodes."""

NUMPY_BOOL_TYPECODES: Final[dict[type[np.generic], str]] = {
    np.bool_: "?",  # bool
}
r"""Dictionary of all `numpy` bool data type typecodes."""

NUMPY_STRING_TYPECODES: Final[dict[type[np.flexible], str]] = {
    np.bytes_: "S",  # str
    np.str_: "U",  # str
    np.void: "V",  # "void"
}
r"""Dictionary of all `numpy` string data type typecodes."""

NUMPY_OBJECT_TYPECODES: Final[dict[type[np.generic], str]] = {
    np.object_: "O",
}
r"""Dictionary of all `numpy` generic data type typecodes."""

NUMPY_TYPECODES: Final[dict[type[np.generic], str]] = (
    NUMPY_INT_TYPECODES
    | NUMPY_UINT_TYPECODES
    | NUMPY_FLOAT_TYPECODES
    | NUMPY_COMPLEX_TYPECODES
    | NUMPY_TIME_TYPECODES
    | NUMPY_STRING_TYPECODES
    | NUMPY_OBJECT_TYPECODES
)
r"""Dictionary of all `numpy` data type typecodes."""

# endregion numpy typecodes --------------------------------------------------------------------------------------------

# region numpy typestrings ---------------------------------------------------------------------------------------------

NUMPY_INT_TYPESTRINGS: Final[dict[type[np.signedinteger], str]] = {
    np.int8: "int8",
    np.int16: "int16",
    np.int32: "int32",
    np.int64: "int64",
    np.byte: "int8",
    np.short: "int16",
    np.intc: "int32",
    np.int_: "int64",
    np.intp: "int64",
    np.longlong: "q",
}
r"""Dictionary of all signed `numpy` integer data type typestrings."""

NUMPY_UINT_TYPESTRINGS: Final[dict[type[np.unsignedinteger], str]] = {
    np.uint8: "uint8",
    np.uint16: "uint16",
    np.uint32: "uint32",
    np.uint64: "uint64",
    np.ubyte: "uint8",
    np.ushort: "uint16",
    np.uintc: "uint32",
    np.uint: "uint64",
    np.uintp: "uint64",
    np.ulonglong: "Q",
}
r"""Dictionary of all unsigned `numpy` integer data type typestrings."""

NUMPY_FLOAT_TYPESTRINGS: Final[dict[type[np.floating], str]] = {
    np.float16: "float16",
    np.float32: "float32",
    np.float64: "float64",
    np.float128: "float128",
    np.half: "float16",
    np.single: "float32",
    np.double: "float64",
    np.longdouble: "float128",
}
r"""Dictionary of all `numpy` float data type typestrings."""

NUMPY_COMPLEX_TYPESTRINGS: Final[dict[type[np.complexfloating], str]] = {
    np.complex64: "complex64",
    np.complex128: "complex128",
    np.complex256: "complex256",
    np.csingle: "complex64",
    np.cdouble: "complex128",
    np.clongdouble: "complex256",
}
r"""Dictionary of all `numpy` complex data typestrings."""


NUMPY_TIME_TYPESTRINGS: Final[dict[type[np.generic], str]] = {
    np.timedelta64: "timedelta64",  # timedelta64
    np.datetime64: "datetime64",  # datetime64
}
r"""Dictionary of all `numpy` time data type typestrings."""

NUMPY_BOOL_TYPESTRINGS: Final[dict[type[np.generic], str]] = {
    np.bool_: "bool",  # bool
}
r"""Dictionary of all `numpy` bool data type typestrings."""

NUMPY_STRING_TYPESTRINGS: Final[dict[type[np.flexible], str]] = {
    np.bytes_: "bytes",  # str
    np.str_: "str",  # str
    np.void: "void",  # "void"
}
r"""Dictionary of all `numpy` string data type typestrings."""

NUMPY_OBJECT_TYPESTRINGS: Final[dict[type[np.generic], str]] = {
    np.object_: "object",
}
r"""Dictionary of all `numpy` generic data type typestrings."""

NUMPY_TYPESTRINGS: Final[dict[type[np.generic], str]] = (
    NUMPY_INT_TYPESTRINGS
    | NUMPY_UINT_TYPESTRINGS
    | NUMPY_FLOAT_TYPESTRINGS
    | NUMPY_COMPLEX_TYPESTRINGS
    | NUMPY_TIME_TYPESTRINGS
    | NUMPY_STRING_TYPESTRINGS
    | NUMPY_OBJECT_TYPESTRINGS
)
r"""Dictionary of all `numpy` data type typestrings."""

# endregion numpy typestrings ------------------------------------------------------------------------------------------


NUMPY_DTYPES: Final[dict[str, type[np.generic]]] = {
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "float128": np.float128,
    "complex64": np.complex64,
    "complex128": np.complex128,
    "complex256": np.complex256,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "timedelta64": np.timedelta64,
    "datetime64": np.datetime64,
    "bool": np.bool_,
    "bytes": np.bytes_,
    "str": np.str_,
    "void": np.void,
    "object": np.object_,
}
r"""Dictionary of all `numpy` data types."""


# region pandas typestrings --------------------------------------------------------------------------------------------

PANDAS_TYPESTRINGS: Final[dict[type[ExtensionDtype], str]] = {
    pandas.BooleanDtype: "boolean",
    pandas.CategoricalDtype: "category",
    pandas.DatetimeTZDtype: "datetime64[ns, tz]",  # datetime64[ns, <tz>]
    pandas.Float32Dtype: "Float32",
    pandas.Float64Dtype: "Float64",
    pandas.Int16Dtype: "Int16",
    pandas.Int32Dtype: "Int32",
    pandas.Int64Dtype: "Int64",
    pandas.Int8Dtype: "Int8",
    pandas.IntervalDtype: "interval",  # e.g. to denote ranges of variables
    pandas.PeriodDtype: "period",  # period[<freq>]
    pandas.SparseDtype: "Sparse",
    pandas.StringDtype: "string",
    pandas.UInt16Dtype: "UInt16",
    pandas.UInt32Dtype: "UInt32",
    pandas.UInt64Dtype: "UInt64",
    pandas.UInt8Dtype: "UInt8",
}
r"""Dictionary of all `pandas` data type typestrings."""

PANDAS_DTYPES: Final[dict[str, type[ExtensionDtype]]] = {
    "boolean": pandas.BooleanDtype,
    "category": pandas.CategoricalDtype,
    "datetime64[ns, tz]": pandas.DatetimeTZDtype,  # datetime64[ns, <tz>]
    "Float32": pandas.Float32Dtype,
    "Float64": pandas.Float64Dtype,
    "Int16": pandas.Int16Dtype,
    "Int32": pandas.Int32Dtype,
    "Int64": pandas.Int64Dtype,
    "Int8": pandas.Int8Dtype,
    "interval": pandas.IntervalDtype,  # e.g. to denote ranges of variables
    "period": pandas.PeriodDtype,  # period[<freq>]
    "Sparse": pandas.SparseDtype,
    "string": pandas.StringDtype,
    "UInt16": pandas.UInt16Dtype,
    "UInt32": pandas.UInt32Dtype,
    "UInt64": pandas.UInt64Dtype,
    "UInt8": pandas.UInt8Dtype,
}
r"""Dictionary of all `pandas` data types."""


# endregion pandas typestrings -----------------------------------------------------------------------------------------


# region torch dtypes --------------------------------------------------------------------------------------------------

TORCH_INT_TYPESTRINGS: Final[dict[torch.dtype, str]] = {
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.qint8: "qint8",
    torch.qint32: "qint32",
}
r"""Dictionary of all `torch` signed integer data type typestrings."""

TORCH_UINT_TYPESTRINGS: Final[dict[torch.dtype, str]] = {
    torch.uint8: "uint8",
    # torch.uint16: "uint16",
    # torch.uint32: "uint32",
    # torch.uint64: "uint64",
    torch.quint8: "quint8",
    torch.quint4x2: "quint4x2",
}
r"""Dictionary of all `torch` unsigned integer data type typestrings."""

TORCH_FLOAT_TYPESTRINGS: Final[dict[torch.dtype, str]] = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
}
r"""Dictionary of all `torch` float data type typestrings."""

TORCH_COMPLEX_TYPESTRINGS: Final[dict[torch.dtype, str]] = {
    torch.complex32: "complex32",
    torch.complex64: "complex64",
    torch.complex128: "complex128",
}
r"""Dictionary of all `torch` complex data type typestrings."""

TORCH_BOOL_TYPESTRINGS: Final[dict[torch.dtype, str]] = {
    torch.bool: "bool",
}
r"""Dictionary of all `torch` bool data type typestrings."""

TORCH_TYPESTRINGS: Final[dict[torch.dtype, str]] = (
    TORCH_INT_TYPESTRINGS
    | TORCH_UINT_TYPESTRINGS
    | TORCH_FLOAT_TYPESTRINGS
    | TORCH_COMPLEX_TYPESTRINGS
    | TORCH_BOOL_TYPESTRINGS
)
r"""Dictionary of all `torch` data type typestrings."""

TORCH_DTYPES: Final[dict[str, torch.dtype]] = {
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "qint8": torch.qint8,
    "qint32": torch.qint32,
    "uint8": torch.uint8,
    "quint8": torch.quint8,
    "quint4x2": torch.quint4x2,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "complex32": torch.complex32,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
    "bool": torch.bool,
}
r"""Dictionary of all `torch` data types."""


# endregion torch dtypes -----------------------------------------------------------------------------------------------

# region python dtypes -------------------------------------------------------------------------------------------------

PYTHON_DTYPES: Final[dict[str, type]] = {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "str": str,
    "bytes": bytes,
    "datetime": datetime,
    "timedelta": timedelta,
    "object": object,
}
r"""Dictionary of all `python` data types."""

PYTHON_TYPESTRINGS: Final[dict[type, str]] = {
    bool: "bool",
    int: "int",
    float: "float",
    complex: "complex",
    str: "str",
    bytes: "bytes",
    datetime: "datetime",
    timedelta: "timedelta",
    object: "object",
}
r"""Dictionary of all `python` data types."""

# endregion python dtypes ----------------------------------------------------------------------------------------------


ScalarDType: TypeAlias = type[np.generic] | torch.dtype | type[ExtensionDtype]  # type: ignore[index]
r"""TypeAlias for scalar types."""

TYPESTRINGS: Final[dict[type[np.generic] | torch.dtype | type[ExtensionDtype], str]] = (
    NUMPY_TYPESTRINGS | TORCH_TYPESTRINGS | PANDAS_TYPESTRINGS  # type: ignore[operator]
)
r"""Dictionary of all type strings."""


PRECISION: Final[dict] = {
    16: 2**-11,
    32: 2**-24,
    64: 2**-53,
    torch.float16: 2**-11,
    torch.float32: 2**-24,
    torch.float64: 2**-53,
    np.float16: 2**-11,
    np.float32: 2**-24,
    np.float64: 2**-53,
}
r"""Maps precision to the corresponding precision factor."""


BOOLS: Final[list[bool]] = [True, False]
r"""List of example bool objects."""

EMOJIS: Final[list[str]] = list(
    "😀😁😂😃😄😅😆😇😈😉😊😋😌😍😎😏"
    "😐😑😒😓😔😕😖😗😘😙😚😛😜😝😞😟"
    "😠😡😢😣😤😥😦😧😨😩😪😫😬😭😮😯"
    "😰😱😲😳😴😵😶😷😸😹😺😻😼😽😾😿"
    "🙀🙁🙂🙃🙄🙅🙆🙇🙈🙉🙊🙋🙌🙍🙎🙏"
)
r"""List of example unicode objects."""


STRINGS: Final[list[str]] = [
    "Alfa",
    "Bravo",
    "Charlie",
    "Delta",
    "Echo",
    "Foxtrot",
    "Golf",
    "Hotel",
    "India",
    "Juliett",
    "Kilo",
    "Lima",
    "Mike",
    "November",
    "Oscar",
    "Papa",
    "Quebec",
    "Romeo",
    "Sierra",
    "Tango",
    "Uniform",
    "Victor",
    "Whiskey",
    "X-ray",
    "Yankee",
    "Zulu",
]
r"""List of example string objects."""


label = namedtuple("label", ["object", "color"])

CATEGORIES: Final[list[label]] = [
    label(object="bear", color="brown"),
    label(object="bear", color="black"),
    label(object="bear", color="white"),
    label(object="beet", color="red"),
    label(object="beet", color="yellow"),
    label(object="beet", color="orange"),
    label(object="beet", color="white"),
    label(object="beet", color="violet"),
]
r"""List of example categorical objects."""
