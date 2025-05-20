"""Implements the BaseConfig class."""

__all__ = [
    # Classes
    "Config",
    # Functions
    "is_allcaps",
    "is_dunder",
    "flatten_dict",
    "unflatten_dict",
]

from abc import ABCMeta
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from typing import Any


def is_allcaps(s: str) -> bool:
    r"""Check if a string is all caps."""
    return s.isidentifier() and s.isupper() and s.isalpha()


def is_dunder(s: str) -> bool:
    r"""Check if a string is a dunder."""
    return s.isidentifier() and s.startswith("__") and s.endswith("__")


class ConfigMetaclass(ABCMeta):  # noqa: B024
    r"""Metaclass for `BaseConfig`."""

    # fmt: off
    _FORBIDDEN_FIELDS = {
        "clear",       # Removes all the elements from the dictionary
        "copy",        # Returns a copy of the dictionary
        "fromkeys",    # Returns a dictionary with the specified keys and value
        "get",         # Returns the value of the specified key
        "items",       # Returns a list containing a tuple for each key value pair
        "keys",        # Returns a list containing the dictionary's keys
        "pop",         # Removes the element with the specified key
        "popitem",     # Removes the last inserted key-value pair
        "setdefault",  # Returns the value of the specified key or set default
        "update",      # Updates the dictionary with the specified key-value pairs
        "values",      # Returns a list of all the values in the dictionary
    }
    # fmt: on

    def __new__(
        cls,
        name: str,
        bases: tuple[type, ...],
        attrs: dict[str, Any],
        **kwds: dict[str, Any],
    ) -> type:
        r"""Create a new class, patch in dataclass fields, and return it."""
        if "__annotations__" not in attrs:
            attrs["__annotations__"] = {}

        config_type = super().__new__(cls, name, bases, attrs, **kwds)
        FIELDS = set(attrs["__annotations__"])

        # check forbidden fields
        FORBIDDEN_FIELDS = cls._FORBIDDEN_FIELDS & FIELDS
        if FORBIDDEN_FIELDS:
            raise ValueError(
                f"Fields '{cls._FORBIDDEN_FIELDS}' are not allowed! "
                f"Found '{FORBIDDEN_FIELDS}'"
            )

        # check for dunder fields
        DUNDER_FIELDS = {key for key in FIELDS if is_dunder(key)}
        if DUNDER_FIELDS:
            raise ValueError(
                f"Dunder fields are not allowed!" f"Found '{DUNDER_FIELDS}'."
            )

        # check all caps fields
        ALLCAPS_FIELDS = {key for key in FIELDS if is_allcaps(key)}
        if ALLCAPS_FIELDS:
            raise ValueError(
                f"ALLCAPS fields are reserved!" f"Found '{ALLCAPS_FIELDS}'."
            )

        NAME = config_type.__qualname__.rsplit(".", maxsplit=1)[0]
        patched_fields = [
            ("_", KW_ONLY),
            ("NAME", str, field(default=NAME)),
            ("MODULE", str, field(default=attrs["__module__"])),
        ]

        for key, hint, *value in patched_fields:
            config_type.__annotations__[key] = hint
            if value:
                setattr(config_type, key, value[0])

        return dataclass(config_type, eq=False, frozen=True)  # type: ignore[call-overload]


class Config(Mapping, metaclass=ConfigMetaclass):
    r"""Base Config."""

    def __iter__(self) -> Iterator[str]:
        r"""Return an iterator over the keys of the dictionary."""
        return iter(self.__dict__)

    def __getitem__(self, key):
        r"""Return the value of the specified key."""
        return self.__dict__[key]

    def __len__(self) -> int:
        r"""Return the number of items in the dictionary."""
        return len(self.__dict__)

    def __hash__(self) -> int:
        r"""Return permutation-invariant hash on `items()`."""
        # Alternative:
        # https://stackoverflow.com/questions/1151658/python-hashable-dicts
        # return hash((frozenset(self), frozenset(self.itervalues())))
        return hash(frozenset(self.items()))

    def __or__(self, other):
        r"""Return a new dictionary with the keys from both dictionaries."""
        res: dict = {}
        res.update(self)
        res.update(other)
        return self.__class__(**res)


def flatten_dict(
    d: dict[str, Any],
    /,
    *,
    recursive: bool = True,
    how: Callable[[Iterable[str]], str] = ".".join,
) -> dict[str, Any]:
    r"""Flatten a dictionary containing iterables to a list of tuples."""
    result = {}
    for key, item in d.items():
        if isinstance(item, dict) and recursive:
            subdict = flatten_dict(item, recursive=True, how=how)
            for subkey, subitem in subdict.items():
                result[how((key, subkey))] = subitem
        else:
            result[key] = item
    return result


def unflatten_dict(
    d: dict[str, Any],
    /,
    *,
    recursive: bool = True,
    how: Callable[[str], Sequence[str]] = str.split,
) -> dict[str, Any]:
    r"""Flatten a dictionary containing iterables to a list of tuples."""
    result = {}
    for key, item in d.items():
        if len(how(key)) > 1 and recursive:
            subdict = unflatten_dict({how(key)[-1]: item}, recursive=True, how=how)
            result[how(key)[0]] = subdict
    return result
