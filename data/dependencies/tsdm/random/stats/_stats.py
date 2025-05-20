r"""Utility functions to get statistics from dataset."""

__all__ = [
    # Functions
    "data_overview",
    "sparsity",
]

import pandas
from pandas import DataFrame, Series


def sparsity(df: DataFrame) -> tuple[float, float]:
    r"""Quantify sparsity in the data."""
    mask = pandas.isna(df)
    col_wise = mask.mean(axis=0)
    total = mask.mean()
    return col_wise, total


# def linearness():
#     r"""Quantify linear signals in the data using regularized least-squares."""
#
#
# def periodicity():
#     r"""Quantify periodic signals in the data using (Non-Uniform) FFT in O(N log N)."""
#
#
# def summary_stats():
#     r"""Summary statistics: column-wise mean/median/std/histogram. Cross-channel correlation."""


def data_overview(df: DataFrame) -> DataFrame:
    r"""Get a summary of the data.

    Parameters
    ----------
    df: DataFrame

    Returns
    -------
    DataFrame
    """
    overview = DataFrame(index=df.columns)
    mask = df.isna()
    overview["# datapoints"] = (~mask).sum()
    overview["% missing"] = (mask.mean() * 100).round(2)
    overview["min"] = df.min().round(2)
    overview["mean"] = df.mean().round(2)
    overview["std"] = df.std().round(2)
    overview["max"] = df.max().round(2)
    freq = {}
    for col in df:
        mask = pandas.notna(df[col].squeeze())
        time = df.index[mask]
        freq[col] = Series(time).diff().mean()
    overview["freq"] = Series(freq)
    return overview
