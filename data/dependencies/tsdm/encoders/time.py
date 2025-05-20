r"""#TODO add module summary line.

#TODO add module description.
"""

from __future__ import annotations

__all__ = [
    # Classes
    "Time2Float",
    "DateTimeEncoder",
    "TimeDeltaEncoder",
]

from typing import Any, Hashable, Literal, Optional, cast

import numpy as np
import pandas as pd
from pandas import DataFrame, DatetimeIndex, Series, Timedelta, Timestamp

from data.dependencies.tsdm.encoders._modular import FrameEncoder
from data.dependencies.tsdm.encoders.base import BaseEncoder


class Time2Float(BaseEncoder):
    r"""Convert `Series` encoded as `datetime64` or `timedelta64` to `floating`.

    By default, the data is mapped onto the unit interval $[0,1]$.
    """

    original_dtype: np.dtype
    offset: Any
    common_interval: Any
    scale: Any

    def __init__(self, normalization: Literal["gcd", "max", "none"] = "max"):
        r"""Choose the normalizations scheme.

        Parameters
        ----------
        normalization: Literal["gcd", "max", "none"], default "max"
        """
        super().__init__()
        self.normalization = normalization

    def fit(self, data: Series, /) -> None:
        r"""Fit to the data.

        Parameters
        ----------
        data: Series
        """
        ds = data
        self.original_dtype = ds.dtype
        self.offset = ds[0].copy()
        assert (
            ds.is_monotonic_increasing
        ), "Time-Values must be monotonically increasing!"

        assert not (
            np.issubdtype(self.original_dtype, np.floating)
            and self.normalization == "gcd"
        ), f"{self.normalization=} illegal when original dtype is floating."

        if np.issubdtype(ds.dtype, np.datetime64):
            ds = ds.view("datetime64[ns]")
            self.offset = ds[0].copy()
            timedeltas = ds - self.offset
        elif np.issubdtype(ds.dtype, np.timedelta64):
            timedeltas = ds.view("timedelta64[ns]")
        elif np.issubdtype(ds.dtype, np.integer):
            timedeltas = ds.view("timedelta64[ns]")
        elif np.issubdtype(ds.dtype, np.floating):
            self.LOGGER.warning("Array is already floating dtype.")
            timedeltas = ds
        else:
            raise ValueError(f"{ds.dtype=} not supported")

        if self.normalization == "none":
            self.scale = np.array(1).view("timedelta64[ns]")
        elif self.normalization == "gcd":
            self.scale = np.gcd.reduce(timedeltas.view(int)).view("timedelta64[ns]")
        elif self.normalization == "max":
            self.scale = ds[-1]
        else:
            raise ValueError(f"{self.normalization=} not understood")

    def encode(self, data: Series, /) -> Series:
        r"""Encode the data.

        Roughly equal to::
            ds ⟶ ((ds - offset)/scale).astype(float)

        Parameters
        ----------
        data: Series

        Returns
        -------
        Series
        """
        if np.issubdtype(data.dtype, np.integer):
            return data
        if np.issubdtype(data.dtype, np.datetime64):
            data = data.view("datetime64[ns]")
            self.offset = data[0].copy()
            timedeltas = data - self.offset
        elif np.issubdtype(data.dtype, np.timedelta64):
            timedeltas = data.view("timedelta64[ns]")
        elif np.issubdtype(data.dtype, np.floating):
            self.LOGGER.warning("Array is already floating dtype.")
            return data
        else:
            raise ValueError(f"{data.dtype=} not supported")

        common_interval = np.gcd.reduce(timedeltas.view(int)).view("timedelta64[ns]")

        return (timedeltas / common_interval).astype(float)

    def decode(self, data: Series, /) -> Series:
        r"""Apply the inverse transformation.

        Roughly equal to:
        ``ds ⟶ (scale*ds + offset).astype(original_dtype)``
        """


class DateTimeEncoder(BaseEncoder):
    r"""Encode DateTime as Float."""

    unit: str = "s"
    r"""The base frequency to convert timedeltas to."""
    base_freq: str = "s"
    r"""The frequency the decoding should be rounded to."""
    offset: Timestamp
    r"""The starting point of the timeseries."""
    kind: type[Series] | type[DatetimeIndex]
    r"""Whether to encode as index of Series."""
    name: Optional[str] = None
    r"""The name of the original Series."""
    dtype: np.dtype
    r"""The original dtype of the Series."""
    freq: Optional[Any] = None
    r"""The frequency attribute in case of DatetimeIndex."""

    def __init__(self, unit: str = "s", base_freq: str = "s"):
        r"""Initialize the parameters.

        Parameters
        ----------
        unit: Optional[str]=None
        """
        super().__init__()
        self.unit = unit
        self.base_freq = base_freq

    def fit(self, data: Series | DatetimeIndex, /) -> None:
        r"""Store the offset."""
        if isinstance(data, Series):
            self.kind = Series
        elif isinstance(data, DatetimeIndex):
            self.kind = DatetimeIndex
        else:
            raise ValueError(f"Incompatible {type(data)=}")

        self.offset = Timestamp(Series(data).iloc[0])
        self.name = str(data.name) if data.name is not None else None
        self.dtype = data.dtype
        if isinstance(data, DatetimeIndex):
            self.freq = data.freq

    def encode(self, data: Series | DatetimeIndex, /) -> Series:
        r"""Encode the input."""
        return (data - self.offset) / Timedelta(1, unit=self.unit)

    def decode(self, data: Series, /) -> Series | DatetimeIndex:
        r"""Decode the input."""
        self.LOGGER.debug("Decoding %s", type(data))
        converted = pd.to_timedelta(data, unit=self.unit)
        datetimes = Series(converted + self.offset, name=self.name, dtype=self.dtype)
        if self.kind == Series:
            return datetimes.round(self.base_freq)
        return DatetimeIndex(  # pylint: disable=no-member
            datetimes, freq=self.freq, name=self.name, dtype=self.dtype
        ).round(self.base_freq)

    def __repr__(self) -> str:
        r"""Return a string representation."""
        return f"{self.__class__.__name__}(unit={self.unit!r})"


class TimeDeltaEncoder(BaseEncoder):
    r"""Encode TimeDelta as Float."""

    unit: str = "s"
    r"""The base frequency to convert timedeltas to."""
    base_freq: str = "s"
    r"""The frequency the decoding should be rounded to."""

    def __init__(self, unit: str = "s", base_freq: str = "s"):
        r"""Initialize the parameters.

        Parameters
        ----------
        unit: Optional[str]=None
        """
        super().__init__()
        self.unit = unit
        self.base_freq = base_freq

    def encode(self, data, /):
        r"""Encode the input."""
        return data / Timedelta(1, unit=self.unit)

    def decode(self, data, /):
        r"""Decode the input."""
        result = data * Timedelta(1, unit=self.unit)
        if hasattr(result, "apply"):
            return result.apply(lambda x: x.round(self.base_freq))
        if hasattr(result, "map"):
            return result.map(lambda x: x.round(self.base_freq))
        return result

    def __repr__(self) -> str:
        r"""Return a string representation."""
        return f"{self.__class__.__name__}(unit={self.unit!r})"


class PeriodicEncoder(BaseEncoder):
    r"""Encode periodic data as sin/cos waves."""

    period: float
    freq: float
    dtype: pd.dtype
    colname: Hashable

    def __init__(self, period: Optional[float] = None) -> None:
        super().__init__()
        self._period = period

    def fit(self, x: Series) -> None:
        r"""Fit the encoder."""
        self.dtype = x.dtype
        self.colname = x.name
        self.period = x.max() + 1 if self._period is None else self._period
        self._period = self.period
        self.freq = 2 * np.pi / self.period

    def encode(self, x: Series) -> DataFrame:
        r"""Encode the data."""
        x = self.freq * (x % self.period)  # ensure 0...N-1
        return DataFrame(
            np.stack([np.cos(x), np.sin(x)]).T,
            columns=[f"cos_{self.colname}", f"sin_{self.colname}"],
        )

    def decode(self, x: DataFrame) -> Series:
        r"""Decode the data."""
        x = np.arctan2(x[f"sin_{self.colname}"], x[f"cos_{self.colname}"])
        x = (x / self.freq) % self.period
        return Series(x, dtype=self.dtype, name=self.colname)

    def __repr__(self):
        r"""Pretty-print."""
        return f"{self.__class__.__name__}({self._period})"


class SocialTimeEncoder(BaseEncoder):
    r"""Social time encoding."""

    level_codes = {
        "Y": "year",
        "M": "month",
        "W": "weekday",
        "D": "day",
        "h": "hour",
        "m": "minute",
        "s": "second",
        "µ": "microsecond",
        "n": "nanosecond",
    }
    original_name: Hashable
    original_dtype: pd.dtype
    original_type: type
    rev_cols: list[str]
    level_code: str
    levels: list[str]

    def __init__(self, levels: str = "YMWDhms") -> None:
        super().__init__()
        self.level_code = levels
        self.levels = [self.level_codes[k] for k in levels]

    def fit(self, x: Series, /) -> None:
        r"""Fit the encoder."""
        self.original_type = type(x)
        self.original_name = x.name
        self.original_dtype = x.dtype
        self.rev_cols = [level for level in self.levels if level != "weekday"]

    def encode(self, x: Series, /) -> DataFrame:
        r"""Encode the data."""
        if isinstance(x, DatetimeIndex):
            res = {level: getattr(x, level) for level in self.levels}
        else:
            res = {level: getattr(x, level) for level in self.levels}
        return DataFrame.from_dict(res)

    def decode(self, x: DataFrame, /) -> Series:
        r"""Decode the data."""
        x = x[self.rev_cols]
        s = pd.to_datetime(x)
        return self.original_type(s, name=self.original_name, dtype=self.original_dtype)

    def __repr__(self) -> str:
        r"""Pretty print."""
        return f"SocialTimeEncoder('{self.level_code}')"


class PeriodicSocialTimeEncoder(SocialTimeEncoder):
    r"""Combines SocialTimeEncoder with PeriodicEncoder using the right frequencies."""

    frequencies = {
        "year": 1,
        "month": 12,
        "weekday": 7,
        "day": 365,
        "hour": 24,
        "minute": 60,
        "second": 60,
        "microsecond": 1000,
        "nanosecond": 1000,
    }
    r"""The frequencies of the used `PeriodicEncoder`."""

    def __new__(cls, levels: str = "YMWDhms") -> PeriodicSocialTimeEncoder:
        r"""Construct a new encoder object."""
        self = super().__new__(cls)
        self.__init__(levels)  # type: ignore[misc]
        column_encoders = {
            level: PeriodicEncoder(period=self.frequencies[level])
            for level in self.levels
        }
        obj = FrameEncoder(column_encoders) @ self
        return cast(PeriodicSocialTimeEncoder, obj)
