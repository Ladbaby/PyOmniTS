r"""Contains helper functions.

Numerical Type Hierarchy:


- object
- datetime, timedelta, datetimeTZ
- interval
- period
- string
    - unicode
    - ascii
    - bytes
- numerical
    - complex
    - float
    - int
    - uint
    - bool
- empty (contains only NA)
"""


__all__ = [
    # Constants
    "BOOLEAN_PAIRS",
    "NA_STRINGS",
    "NA_VALUES",
    # Functions
    "get_uniques",
    "string_is_bool",
    "string_to_bool",
    "numeric_is_bool",
    "float_is_int",
]

from typing import Final, Optional, cast

import numpy as np
import pandas
from pandas import Series

NA_STRINGS: Final[set[str]] = {
    r"",
    r"-",
    r"n/a",
    r"N/A",
    r"<na>",
    r"<NA>",
    r"nan",
    r"NaN",
    r"NAN",
    r"NaT",
    r"none",
    r"None",
    r"NONE",
}
r"""String that correspond to NA values."""

NA_VALUES: Final[set] = {
    None,
    float("nan"),
    np.nan,
    pandas.NA,
    pandas.NaT,
    np.datetime64("NaT"),
}
r"""Values that correspond to NaN."""

BOOLEAN_PAIRS: Final[list[dict[str | int | float, bool]]] = [
    {"f": False, "t": True},
    {"false": False, "true": True},
    {"n": False, "y": True},
    {"no": False, "yes": True},
    {"-": False, "+": True},
    {0: False, 1: True},
    {-1: False, +1: True},
    {0.0: False, 1.0: True},
    {-1.0: False, +1.0: True},
]
r"""Matched pairs of values that correspond to booleans."""

# def infer_dtype(series: Series) -> Union[None, ExtensionDtype, np.generic]:
#     original_series = series.copy()
#     inferred_series = series.copy().convert_dtypes()
#     original_dtype = original_series.dtype
#     inferred_dtype = inferred_series.dtype
#
#     series = inferred_series
#     mask = pandas.notna(series)
#
#     # Series contains only NaN values...
#     if not mask.any():
#         return None
#
#     values = series[mask]
#     uniques = values.unique()
#
#     # if string do string downcast
#     if pandas.api.types.is_string_dtype(series):
#         if string_is_bool(series, uniques=uniques):
#             string_to_bool(series, uniques=uniques)


def get_uniques(series: Series, /, *, ignore_nan: bool = True) -> Series:
    r"""Return unique values, excluding nan.

    Parameters
    ----------
    series: Series
    ignore_nan: bool = True

    Returns
    -------
    Series
    """
    if ignore_nan:
        mask = pandas.notna(series)
        series = series[mask]
    return Series(series.unique())


def string_is_bool(series: Series, /, *, uniques: Optional[Series] = None) -> bool:
    r"""Test if 'string' series could possibly be boolean.

    Parameters
    ----------
    series: Series
    uniques: Optional[Series]

    Returns
    -------
    bool
    """
    assert pandas.api.types.is_string_dtype(series), "Series must be 'string' dtype!"
    values = get_uniques(series) if uniques is None else uniques

    if len(values) == 0 or len(values) > 2:
        return False
    return any(
        set(values.str.lower()) <= bool_pair.keys() for bool_pair in BOOLEAN_PAIRS
    )


def string_to_bool(series: Series, uniques: Optional[Series] = None) -> Series:
    r"""Convert Series to nullable boolean.

    Parameters
    ----------
    series: Series
    uniques: Optional[Series]

    Returns
    -------
    bool
    """
    assert pandas.api.types.is_string_dtype(series), "Series must be 'string' dtype!"
    mask = pandas.notna(series)
    values = get_uniques(series[mask]) if uniques is None else uniques
    mapping = next(
        set(values.str.lower()) <= bool_pair.keys() for bool_pair in BOOLEAN_PAIRS
    )
    series = series.copy()
    series[mask] = series[mask].map(mapping)
    return series.astype(pandas.BooleanDtype())


def numeric_is_bool(series: Series, uniques: Optional[Series] = None) -> bool:
    r"""Test if 'numeric' series could possibly be boolean.

    Parameters
    ----------
    series: Series
    uniques: Optional[Series]

    Returns
    -------
    bool
    """
    assert pandas.api.types.is_numeric_dtype(series), "Series must be 'numeric' dtype!"
    values = get_uniques(series) if uniques is None else uniques
    if len(values) == 0 or len(values) > 2:
        return False
    return any(set(values) <= set(bool_pair) for bool_pair in BOOLEAN_PAIRS)


def float_is_int(series: Series, uniques: Optional[Series] = None) -> bool:
    r"""Check whether float encoded column holds only integers.

    Parameters
    ----------
    series: Series
    uniques: Optional[Series] = None

    Returns
    -------
    bool
    """
    assert pandas.api.types.is_float_dtype(series), "Series must be 'float' dtype!"
    values = get_uniques(series) if uniques is None else uniques
    return cast(bool, values.apply(float.is_integer).all())


#
# def get_integer_cols(df) -> set[str]:
#     cols = set()
#     for col in table:
#         if np.issubdtype(table[col].dtype, np.integer):
#             print(f"Integer column                       : {col}")
#             cols.add(col)
#         elif np.issubdtype(table[col].dtype, np.floating) and float_is_int(table[col]):
#             print(f"Integer column pretending to be float: {col}")
#             cols.add(col)
#     return cols
#
#
# def contains_nan_slice(series, slices, two_enough: bool = False) -> bool:
#     num_missing = 0
#     for idx in slices:
#         if pd.isna(series[idx]).all():
#             num_missing += 1
#
#     if (num_missing > 0 and not two_enough) or (
#         num_missing >= len(slices) - 1 and two_enough
#     ):
#         print(f"{series.name}: data missing in {num_missing}/{len(slices)} slices!")
#         return True
#     return False
