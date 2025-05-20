r"""Utility functions for random number generation."""

__all__ = [
    # Functions
    "random_data",
    "sample_timestamps",
    "sample_timedeltas",
]


import logging
from typing import Optional

import numpy as np
from numpy.typing import DTypeLike, NDArray
from pandas import Timedelta, Timestamp, date_range, timedelta_range

from data.dependencies.tsdm.utils.types.dtypes import BOOLS, EMOJIS, STRINGS
from data.dependencies.tsdm.utils.types.time import DTVar, TDVar

__logger__ = logging.getLogger(__name__)


def sample_timestamps(
    start: str | DTVar = "today",
    final: Optional[DTVar] = None,
    *,
    size: int,
    freq: str | TDVar = "1s",
    replace: bool = False,
    include_start: bool = True,
    include_final: bool = False,
) -> NDArray:
    r"""Create randomly sampled timestamps.

    Parameters
    ----------
    start: TimeStampLike, default <today>
    final: TimeStampLike, default <today>+<24h>
    size: int
    freq: TimeDeltaLike, default "1s"
        The smallest possible timedelta between distinct timestamps.
    replace: bool, default True
        Whether the sample is with or without replacement.
    include_start: bool, default True
        If `True`, then `start` will always be the first sampled timestamp.
    include_final: bool, default True
        If `True`, then `final` will always be the final sampled timestamp.

    Returns
    -------
    NDArray
    """
    start_dt = Timestamp(start)
    final_dt = start_dt + Timedelta("24h") if final is None else Timestamp(final)
    freq_td = Timedelta(freq)
    start_dt, final_dt = start_dt.round(freq_td), final_dt.round(freq_td)

    # randomly sample timestamps
    rng = np.random.default_rng()
    timestamps = date_range(start_dt, final_dt, freq=freq_td)
    timestamps = rng.choice(
        timestamps[include_start : -include_final or None],
        size - include_start - include_final,
        replace=replace,
    )
    timestamps = np.sort(timestamps)

    # add boundary if requested
    if include_start:
        timestamps = np.insert(timestamps, 0, start_dt)
    if include_final:
        timestamps = np.insert(timestamps, -1, final_dt)

    # Convert to base unit based on freq
    units: dict[str, np.timedelta64] = {
        u: np.timedelta64(1, u)
        for u in ("Y", "M", "W", "D", "h", "m", "s", "us", "ns", "ps", "fs", "as")
    }
    base_unit = next(u for u, val in units.items() if freq_td >= val)
    return timestamps.astype(f"datetime64[{base_unit}]")


def sample_timedeltas(
    low: str | TDVar = "0s",
    high: str | TDVar = "1h",
    size: Optional[int] = None,
    freq: str | TDVar = "1s",
) -> NDArray:
    r"""Create randomly sampled timedeltas.

    Parameters
    ----------
    low:  TimeDeltaLike, optional
    high: TimeDeltaLike, optional
    size: int,           optional
    freq: TimeDeltaLike, optional

    Returns
    -------
    NDArray
    """
    low_dt = Timedelta(low)
    high_dt = Timedelta(high)
    freq_dt = Timedelta(freq)
    low_dt, high_dt = low_dt.round(freq_dt), high_dt.round(freq_dt)

    # randomly sample timedeltas
    rng = np.random.default_rng()
    timedeltas = timedelta_range(low_dt, high_dt, freq=freq_dt)
    timedeltas = rng.choice(timedeltas, size=size)

    # Convert to base unit based on freq
    units = {
        u: np.timedelta64(1, u)
        for u in ("Y", "M", "W", "D", "h", "m", "s", "us", "ns", "ps", "fs", "as")
    }
    base_unit = next(u for u, val in units.items() if freq_dt >= val)
    return timedeltas.astype(f"timedelta64[{base_unit}]")


def random_data(
    size: tuple[int], dtype: DTypeLike = float, missing: float = 0.5
) -> NDArray:
    r"""Create random data of given size and dtype.

    Parameters
    ----------
    size
    dtype
    missing

    Returns
    -------
    NDArray
    """
    dtype = np.dtype(dtype)
    rng = np.random.default_rng()
    if np.issubdtype(dtype, np.integer):
        iinfo = np.iinfo(dtype)
        data = rng.integers(low=iinfo.min, high=iinfo.max, size=size)
        result = data.astype(dtype)
    elif np.issubdtype(dtype, np.floating):
        finfo = np.finfo(dtype)  # type: ignore[arg-type]
        exp = rng.integers(low=finfo.minexp, high=finfo.maxexp, size=size)
        mant = rng.uniform(low=-2, high=+2, size=size)
        result = (mant * 2**exp).astype(dtype)
    elif np.issubdtype(dtype, np.bool_):
        result = rng.choice(BOOLS, size=size)
    elif np.issubdtype(dtype, np.unicode_):
        result = rng.choice(EMOJIS, size=size)
    elif np.issubdtype(dtype, np.string_):
        result = rng.choice(STRINGS, size=size)
    else:
        raise NotImplementedError

    __logger__.warning("TODO: implement missing%s!", missing)

    return result
