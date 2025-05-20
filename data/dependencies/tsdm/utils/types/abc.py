r"""Type Variables from collections.abc objects."""

__all__ = [
    "ContainerType",
    "HashableType",
    "IterableType",
    "IteratorType",
    "ReversibleType",
    "GeneratorType",
    "SizedType",
    "CallableType",
    "CollectionType",
    "SequenceType",
    "MutableSequenceType",
    "ByteStringType",
    "SetType",
    "MutableSetType",
    "MappingType",
    "MutableMappingType",
    "MappingViewType",
    "ItemsViewType",
    "KeysViewType",
    "ValuesViewType",
    "AwaitableType",
    "CoroutineType",
    "AsyncIterableType",
    "AsyncIteratorType",
    "AsyncGeneratorType",
]

from collections import abc
from typing import TypeVar

#: blah blah blah
ContainerType = TypeVar("ContainerType", bound=abc.Container)
r"""Type variable for Containers."""

#: blah blah blah
HashableType = TypeVar("HashableType", bound=abc.Hashable)
r"""Type variable for Hashable objects."""

IterableType = TypeVar("IterableType", bound=abc.Iterable)
r"""Type variable for Iterables."""

IteratorType = TypeVar("IteratorType", bound=abc.Iterator)
r"""Type variable for Iterators."""

ReversibleType = TypeVar("ReversibleType", bound=abc.Reversible)
r"""Type variable for Reversible."""

GeneratorType = TypeVar("GeneratorType", bound=abc.Generator)
r"""Type variable for Generators."""

SizedType = TypeVar("SizedType", bound=abc.Sized)
r"""Type variable for Mappings."""

CallableType = TypeVar("CallableType", bound=abc.Callable)
r"""Type variable for Callables."""

CollectionType = TypeVar("CollectionType", bound=abc.Collection)
r"""Type variable for Collections."""

SequenceType = TypeVar("SequenceType", bound=abc.Sequence)
r"""Type variable for Sequences."""

MutableSequenceType = TypeVar("MutableSequenceType", bound=abc.MutableSequence)
r"""Type variable for MutableSequences."""

ByteStringType = TypeVar("ByteStringType", bound=abc.ByteString)
r"""Type variable for ByteStrings."""

SetType = TypeVar("SetType", bound=abc.Set)
r"""Type variable for Sets."""

MutableSetType = TypeVar("MutableSetType", bound=abc.MutableSet)
r"""Type variable for MutableSets."""

MappingType = TypeVar("MappingType", bound=abc.Mapping)
r"""Type variable for Mappings."""

MutableMappingType = TypeVar("MutableMappingType", bound=abc.MutableMapping)
r"""Type variable for MutableMappings."""

# Views

MappingViewType = TypeVar("MappingViewType", bound=abc.MappingView)
r"""Type variable for MappingViews."""

ItemsViewType = TypeVar("ItemsViewType", bound=abc.ItemsView)
r"""Type variable for ItemsViews."""

KeysViewType = TypeVar("KeysViewType", bound=abc.KeysView)
r"""Type variable for KeysViews."""

ValuesViewType = TypeVar("ValuesViewType", bound=abc.ValuesView)
r"""Type variable for ValuesViews."""


# Async stuff

AwaitableType = TypeVar("AwaitableType", bound=abc.Awaitable)
r"""Type variable for Awaitables."""

CoroutineType = TypeVar("CoroutineType", bound=abc.Coroutine)
r"""Type variable for Coroutines."""

AsyncIterableType = TypeVar("AsyncIterableType", bound=abc.AsyncIterable)
r"""Type variable for AsyncIterables."""

AsyncIteratorType = TypeVar("AsyncIteratorType", bound=abc.AsyncIterator)
r"""Type variable for AsyncIterators."""

AsyncGeneratorType = TypeVar("AsyncGeneratorType", bound=abc.AsyncGenerator)
r"""Type variable for AsyncGenerators."""
