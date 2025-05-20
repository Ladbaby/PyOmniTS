r"""Utility functions for string manipulation."""

# from __future__ import annotations
#
# from __future__ import annotations

__all__ = [
    # Functions
    "snake2camel",
    # "camel2snake",
    "repr_array",
    "repr_mapping",
    "repr_sequence",
    "repr_namedtuple",
    "tensor_info",
    "dict2string",
]
__ALL__ = dir() + __all__


import builtins
from collections.abc import Callable, Iterable, Mapping, Sequence, Sized
from typing import Any, NamedTuple, Optional, cast, overload

from pandas import DataFrame
from torch import Tensor

from data.dependencies.tsdm.utils.types.dtypes import TYPESTRINGS, ScalarDType
from data.dependencies.tsdm.utils.types.protocols import Array, NTuple


def __dir__() -> list[str]:
    return __ALL__


@overload
def snake2camel(s: str) -> str:
    ...


@overload
def snake2camel(s: list[str]) -> list[str]:
    ...


@overload
def snake2camel(s: tuple[str, ...]) -> tuple[str, ...]:
    ...


def snake2camel(s):
    r"""Convert ``snake_case`` to ``CamelCase``.

    Parameters
    ----------
    s: str | Iterable[str]

    Returns
    -------
    str | Iterable[str]
    """
    if isinstance(s, tuple):
        return tuple(snake2camel(x) for x in s)

    if isinstance(s, Iterable) and not isinstance(s, str):
        return [snake2camel(x) for x in s]

    if isinstance(s, str):
        substrings = s.split("_")
        return "".join(s[0].capitalize() + s[1:] for s in substrings)

    raise TypeError(f"Type {type(s)} nor understood, expected string or iterable.")


def tensor_info(x: Tensor) -> str:
    r"""Print useful information about Tensor."""
    return f"{x.__class__.__name__}[{tuple(x.shape)}, {x.dtype}, {x.device.type}]"


def dict2string(d: dict[str, Any]) -> str:
    r"""Return pretty string representation of dictionary.

    Vertically aligns keys.

    Parameters
    ----------
    d: dict[str, Any]

    Returns
    -------
    str
    """
    max_key_length = max((len(key) for key in d), default=0)
    pad = " " * 2

    string = "dict(" + "\n"

    for key, value in sorted(d.items()):
        string += f"\n{pad}{key:<{max_key_length}}: {repr(value)}"

    string += "\n)"
    return string


def repr_object(obj: Any, **kwargs: Any) -> str:
    r"""Return a string representation of an object.

    Parameters
    ----------
    obj: Any

    Returns
    -------
    str
    """
    if type(obj).__name__ in dir(builtins):
        return str(obj)
    if isinstance(obj, Tensor):
        return repr_array(obj, **kwargs)
    if isinstance(obj, Mapping):
        return repr_mapping(obj, **kwargs)
    if isinstance(obj, NTuple):
        obj = cast(NamedTuple, obj)
        return repr_namedtuple(obj, **kwargs)
    if isinstance(obj, Sequence):
        return repr_sequence(obj, **kwargs)
    try:
        return repr(obj)
    # Fallback Option
    except Exception:
        return repr(type(obj))


def repr_mapping(
    obj: Mapping,
    *,
    linebreaks: bool = True,
    maxitems: int = 6,
    padding: int = 4,
    recursive: bool | int = True,
    repr_fun: Callable[..., str] = repr_object,
    title: Optional[str] = None,
    align: bool = False,
) -> str:
    r"""Return a string representation of a mapping object.

    Parameters
    ----------
    obj: Mapping
    linebreaks: bool, default True
    maxitems: int, default 6
    padding: int
    recursive: bool, default True
    repr_fun: Callable[..., str], default repr_object
    title: Optional[str], default None,
    align:
        Whether to vertically align keys.

    Returns
    -------
    str
    """
    br = "\n" if linebreaks else ""
    # key_sep = ": "
    sep = "," if linebreaks else ", "
    pad = " " * padding * linebreaks

    keys = [str(key) for key in obj.keys()]
    max_key_length = max(len(key) for key in keys) if align else 0

    items = list(obj.items())
    title = type(obj).__name__ if title is None else title
    string = title + "(" + br

    # TODO: automatic linebreak detection if string length exceeds max_length

    def to_string(x: Any) -> str:
        if recursive:
            if isinstance(recursive, bool):
                return repr_fun(x).replace("\n", br + pad)
            return repr_fun(x, recursive=recursive - 1).replace("\n", br + pad)
        return repr_type(x)

    # keys = [str(key) for key in obj.keys()]
    # values: list[str] = [to_string(x) for x in obj.values()]

    if len(obj) <= maxitems:
        string += "".join(
            f"{pad}{str(key):<{max_key_length}}: {to_string(value)}{sep}{br}"
            for key, value in items
        )
    else:
        string += "".join(
            f"{pad}{str(key):<{max_key_length}}: {to_string(value)}{sep}{br}"
            for key, value in items[: maxitems // 2]
        )
        string += f"{pad}...\n"
        string += "".join(
            f"{pad}{str(key):<{max_key_length}}: {to_string(value)}{sep}{br}"
            for key, value in items[-maxitems // 2 :]
        )
    string += ")"
    return string


def repr_sequence(
    obj: Sequence,
    *,
    linebreaks: bool = True,
    maxitems: int = 6,
    padding: int = 4,
    recursive: bool | int = True,
    repr_fun: Callable[..., str] = repr_object,
    title: Optional[str] = None,
) -> str:
    r"""Return a string representation of a sequence object.

    Parameters
    ----------
    obj: Sequence
    linebreaks: bool, default True
    maxitems: int, default 6
    padding: int
    recursive: bool, default True
    repr_fun: Callable[..., str], default repr_object
    title: Optional[str], default None,

    Returns
    -------
    str
    """
    br = "\n" if linebreaks else ""
    sep = "," if linebreaks else ", "
    pad = " " * padding * linebreaks
    title = type(obj).__name__ if title is None else title
    string = title + "(" + br

    def to_string(x: Any) -> str:
        if recursive:
            if isinstance(recursive, bool):
                return repr_fun(x).replace("\n", br + pad)
            return repr_fun(x, recursive=recursive - 1).replace("\n", br + pad)
        return repr_type(x)

    if len(obj) <= maxitems:
        string += "".join(f"{pad}{to_string(value)}{sep}{br}" for value in obj)
    else:
        string += "".join(
            f"{pad}{to_string(value)}{sep}{br}" for value in obj[: maxitems // 2]
        )
        string += f"{pad}..." + br
        string += "".join(
            f"{pad}{to_string(value)}{sep}{br}" for value in obj[-maxitems // 2 :]
        )
    string += ")"

    return string


def repr_namedtuple(
    obj: NamedTuple,
    *,
    linebreaks: bool = True,
    maxitems: int = 6,
    padding: int = 4,
    recursive: bool | int = True,
    repr_fun: Callable[..., str] = repr_object,
    title: Optional[str] = None,
) -> str:
    r"""Return a string representation of a namedtuple object.

    Parameters
    ----------
    obj: tuple
    linebreaks: bool, default True
    maxitems: int, default 6
    padding: int
    recursive: bool | int, default True
    repr_fun: Callable[..., str], default repr_object
    title: Optional[str], default None,

    Returns
    -------
    str
    """
    title = type(obj).__name__ if title is None else title

    # if not hasattr(obj, "_asdict"):

    return repr_mapping(
        obj._asdict(),
        padding=padding,
        maxitems=maxitems,
        title=title,
        repr_fun=repr_fun,
        linebreaks=linebreaks,
        recursive=recursive,
    )


def repr_array(obj: Array | DataFrame, *, title: Optional[str] = None) -> str:
    r"""Return a string representation of an array object.

    Parameters
    ----------
    obj: ArrayLike
    title: Optional[str] = None

    Returns
    -------
    str
    """
    assert isinstance(obj, Array)

    title = type(obj).__name__ if title is None else title

    string = title + "["
    string += str(tuple(obj.shape))

    if isinstance(obj, DataFrame):
        dtypes = [repr_dtype(dtype) for dtype in obj.dtypes]
        string += ", " + repr_sequence(dtypes, linebreaks=False)
    elif isinstance(obj, Array):
        string += ", " + repr_dtype(obj.dtype)
    else:
        raise TypeError(f"Cannot get dtype of {type(obj)}")
    if isinstance(obj, Tensor):
        string += f"@{obj.device}"

    string += "]"
    return string


def repr_sized(obj: Sized, *, title: Optional[str] = None) -> str:
    r"""Return a string representation of a sized object.

    Parameters
    ----------
    obj: Sized
    title: Optional[str], default None

    Returns
    -------
    str
    """
    title = type(obj).__name__ if title is None else title
    string = title + "["
    string += str(len(obj))
    string += "]"
    return string


def repr_dtype(dtype: str | ScalarDType) -> str:
    r"""Return a string representation of a dtype object.

    Parameters
    ----------
    dtype: str | ScalarDtype | ExtensionDtype

    Returns
    -------
    str
    """
    if isinstance(dtype, str):
        return dtype
    if dtype in TYPESTRINGS:
        return TYPESTRINGS[dtype]
    return str(dtype)


def repr_type(obj: Any) -> str:
    r"""Return a string representation of an object.

    Parameters
    ----------
    obj: Any

    Returns
    -------
    str
    """
    if isinstance(obj, Array):
        return repr_array(obj)
    if isinstance(obj, Sized):
        return repr_sized(obj)
    if isinstance(obj, type):
        return obj.__name__
    return obj.__class__.__name__ + "()"
