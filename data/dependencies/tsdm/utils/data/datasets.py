r"""Generic Dataset classes."""

from __future__ import annotations

__all__ = [
    # Classes
    "MappingDataset",
    "DatasetCollection",
]

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, Optional

from pandas import DataFrame, MultiIndex
from torch.utils.data import Dataset

from data.dependencies.tsdm.utils.strings import repr_mapping


class MappingDataset(Dataset, Mapping):
    r"""Represents a Mapping[Key, Dataset].

    ``ds[key]`` returns the dataset for the given key.
    If the key is a tuple, tries to divert to the nested dataset.

    ``ds[(key, subkey)]=ds[key][subkey]``
    """

    def __init__(self, data: Mapping[Any, Dataset]):
        r"""Initialize the dataset.

        Parameters
        ----------
        data: Mapping
        """
        super().__init__()
        assert isinstance(data, Mapping)
        if isinstance(data, Mapping):
            self.index = list(data.keys())
            self.data = data

    def __iter__(self) -> Iterator:
        r"""Iterate over the keys."""
        return iter(self.index)

    def __len__(self) -> int:
        r"""Length of the dataset."""
        return len(self.index)

    def __getitem__(self, key):
        r"""Get the dataset for the given key."""
        if not isinstance(key, tuple):
            return self.data[key]
        try:
            outer = self.data[key[0]]
            return outer[key[1:]]
        except KeyError:
            return self.data[key]

    @staticmethod
    def from_dataframe(
        df: DataFrame, levels: Optional[list[str]] = None
    ) -> MappingDataset:
        r"""Create a MappingDataset from a DataFrame.

        Parameters
        ----------
        df: DataFrame
        levels: Optional[list[str]]
            If given, the selected levels from the DataFrame's MultiIndex are used as keys.

        Returns
        -------
        MappingDataset
        """
        if levels is not None:
            min_index = df.index.to_frame()
            sub_index = MultiIndex.from_frame(min_index[levels])
            index = sub_index.unique()
        else:
            index = df.index

        return MappingDataset({idx: df.loc[idx] for idx in index})

    def __repr__(self):
        r"""Representation of the dataset."""
        return repr_mapping(self)


class DatasetCollection(Dataset, Mapping):
    r"""Represents a ``mapping[index → torch.Datasets]``.

    All tensors must have a shared index,
    in the sense that index.unique() is identical for all inputs.
    """

    dataset: dict[Any, Dataset]
    r"""The dataset."""

    def __init__(self, indexed_datasets: Mapping[Any, Dataset]):
        super().__init__()
        self.dataset = dict(indexed_datasets)
        self.index = list(self.dataset.keys())
        self.keys = self.dataset.keys  # type: ignore[assignment]
        self.values = self.dataset.values  # type: ignore[assignment]
        self.items = self.dataset.items  # type: ignore[assignment]

    def __len__(self):
        r"""Length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, item):
        r"""Hierarchical lookup."""
        # test for hierarchical indexing
        if isinstance(item, Sequence):
            first, rest = item[0], item[1:]
            if isinstance(first, (Iterable, slice)):
                # pass remaining indices to sub-object
                value = self.dataset[first]
                return value[rest]

        # no hierarchical indexing
        return self.dataset[item]

    def __iter__(self):
        r"""Iterate over the dataset."""
        for key in self.index:
            yield self.dataset[key]

    def __repr__(self):
        r"""Representation of the dataset."""
        return repr_mapping(self)


# class IterItems(TorchDataset):
#     r"""A thin wrapper around a dataset that yields items from the dataset."""
#
#     def __init__(self, dataset: TorchDataset) -> None:
#         super().__init__()
#         self.dataset = dataset
#
#     def __getitem__(self, key: Any) -> tuple[Any, Any]:
#         r"""Get the item from the dataset."""
#         return Items(key, self.dataset[key])
#
#     def __repr__(self) -> str:
#         r"""Representation of the dataset."""
#         return r"IterItems@" + self.dataset.__repr__()
#
#     def __getattr__(self, item):
#         r"""Forward all other attributes to the dataset."""
#         return getattr(self.dataset, item)
#
#     def __iter__(self):
#         r"""Forward to wrapped object."""
#
# from collections import namedtuple
# class TupleDataset(Dataset[tuple[Tensor, ...]]):
#     r"""Sequential Dataset."""
#
#     def __init__(self, tensors: tuple[Tensor, ...] | dict[str, Tensor], /, *, index: Optional[Series] = None):
#         assert all(len(x) == len(tensors[0]) for x in tensors)
#         self.tensors = tensors
#
#     def __len__(self):
#         r"""Length of the dataset."""
#         return len(self.tensors[0])
#
#     def __getitem__(self, idx) -> tuple[Tensor, ...]:
#         r"""Get the same slice from each tensor."""
#         return tuple(x[idx] for x in self.tensors)
