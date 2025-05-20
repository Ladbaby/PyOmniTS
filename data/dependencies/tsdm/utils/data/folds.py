r"""Utility function to process folds and splits."""

__all__ = [
    # functions
    "folds_as_frame",
    "folds_as_sparse_frame",
    "folds_from_groups",
]

from collections.abc import Mapping, Sequence
from typing import Optional, TypeAlias

import numpy as np
from pandas import DataFrame, Index, MultiIndex, Series

FOLDS: TypeAlias = Sequence[Mapping[str, Series]]
r"""Type Hint for Folds. the series must be boolean."""


def folds_from_groups(
    groups: Series, /, *, num_folds: int = 5, seed: Optional[int] = None, **splits: int
) -> FOLDS:
    r"""Create folds from a Series of groups.

    Parameters
    ----------
    groups: Series[int]
        Series of group labels.
    num_folds: int
        Number of folds to create.
    seed: int
        Seed for the random number generator.
    splits: int
        Relative number of samples in each split.
        E.g. ``folds_from_groups(groups, train=7, valid=2, test=1)`` uses 7/10 of the
        samples for training, 2/10 for validation and 1/10 for testing.

    Returns
    -------
    folds: FOLDS

    This is useful, when the data needs to be grouped, e.g. due to replicate experiments.
    Simply use `pandas.groupby` and pass the result to this function.
    """
    assert splits, "No splits provided"
    num_chunks = sum(splits.values())
    q, remainder = divmod(num_chunks, num_folds)
    assert remainder == 0, "Sum of chunks must be a multiple of num_folds"

    unique_groups = groups.unique()
    generator = np.random.default_rng(seed)
    shuffled = generator.permutation(unique_groups)
    chunks = np.array(np.array_split(shuffled, num_chunks), dtype=object)

    slices, a, b = {}, 0, 0
    for key, size in splits.items():
        a, b = b, b + size
        slices[key] = np.arange(a, b)

    folds = []
    for k in range(num_folds):
        fold = {}
        for key in splits:
            mask = (slices[key] + q * k) % num_chunks
            selection = chunks[mask]
            chunk = np.concatenate(selection)
            fold[key] = groups.isin(chunk)
        folds.append(fold)

    return folds


def folds_as_frame(
    folds: FOLDS, /, *, index: Optional[Sequence] = None, sparse: bool = False
) -> DataFrame:
    r"""Create a table holding the fold information.

    +-------+-------+-------+-------+-------+-------+
    | fold  | 0     | 1     | 2     | 3     | 4     |
    +=======+=======+=======+=======+=======+=======+
    | 15325 | train | train |  test | train | train |
    +-------+-------+-------+-------+-------+-------+
    | 15326 | train | train | train | train |  test |
    +-------+-------+-------+-------+-------+-------+
    | 15327 |  test | valid | train | train | train |
    +-------+-------+-------+-------+-------+-------+
    | 15328 | valid | train | train | train |  test |
    +-------+-------+-------+-------+-------+-------+
    """
    if index is None:
        # create a default index
        first_fold = next(iter(folds))
        first_split = next(iter(first_fold.values()))

        name_index: Sequence = (
            first_split.index
            if isinstance(first_split, Series)
            else np.arange(len(first_split))
        )
    else:
        name_index = index

    fold_idx = Index(range(len(folds)), name="fold")
    splits = DataFrame(index=name_index, columns=fold_idx, dtype="string")

    for k in fold_idx:
        for key, split in folds[k].items():
            # where cond is false is replaces with key
            splits[k] = splits[k].where(~split, key)

    if not sparse:
        return splits

    return folds_as_sparse_frame(splits)


def folds_as_sparse_frame(df: DataFrame, /) -> DataFrame:
    r"""Create a sparse table holding the fold information."""
    # TODO: simplify this code. It should just be pd.concat(folds)
    columns = df.columns

    # get categoricals
    categories = {col: df[col].astype("category").dtype.categories for col in columns}

    if isinstance(df.columns, MultiIndex):
        index_tuples = [
            (*col, cat)
            for col, cats in zip(columns, categories)
            for cat in categories[col]
        ]
        names = df.columns.names + ["partition"]
    else:
        index_tuples = [
            (col, cat)
            for col, cats in zip(columns, categories)
            for cat in categories[col]
        ]
        names = [df.columns.name, "partition"]

    new_columns = MultiIndex.from_tuples(index_tuples, names=names)
    result = DataFrame(index=df.index, columns=new_columns, dtype=bool)

    if isinstance(df.columns, MultiIndex):
        for col in new_columns:
            result[col] = df[col[:-1]] == col[-1]
    else:
        for col in new_columns:
            result[col] = df[col[0]] == col[-1]

    return result
