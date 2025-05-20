r"""Types related to time."""


__all__ = [
    "DTVar",
    "TDVar",
    "TimeVar",
    "RealDTVar",
    "RealTDVar",
    "RealTimeVar",
    "DT",
    "TD",
    "Time",
    "RealDT",
    "RealTD",
    "RealTime",
    "NumpyTDVar",
    "NumpyTimeVar",
    "NumpyDTVar",
]

from datetime import datetime as py_dt
from datetime import timedelta as py_td
from typing import TypeAlias, TypeVar, Union

from numpy import datetime64 as np_dt
from numpy import floating as np_float
from numpy import integer as np_int
from numpy import timedelta64 as np_td
from pandas import Timedelta as pd_td
from pandas import Timestamp as pd_dt

# TODO: Use TypeAlias Once Python 3.10 comes out.


# Time-Type-Variables
# DTVar = TypeVar("DTVar", bound=Union[int, float, np_int, np_float, py_dt, np_dt, pd_dt])
# r"""TypeVar for `Timestamp` values."""
#
# TDVar = TypeVar("TDVar", bound=Union[int, float, np_int, np_float, py_td, np_td, pd_td])
# r"""TypeVar for `Timedelta` values."""
#
# TimeVar = TypeVar(
#     "TimeVar", bound=Union[int, float, np_int, np_float, py_dt, np_dt, pd_dt, py_td, np_td, pd_td]
# )
# r"""TypeVar for `Time` values."""

DTVar = TypeVar("DTVar", int, float, np_int, np_float, np_dt, pd_dt)
r"""TypeVar for `Timestamp` values."""

TDVar = TypeVar("TDVar", int, float, np_int, np_float, np_td, pd_td)
r"""TypeVar for `Timedelta` values."""

TimeVar = TypeVar("TimeVar", int, float, np_int, np_float, np_dt, pd_dt, np_td, pd_td)
r"""TypeVar for `Time` values."""

# Time-Type-Hints
DT: TypeAlias = Union[int, float, np_int, np_float, py_dt, np_dt, pd_dt]
r"""Type Hint for `Timestamp`."""
TD: TypeAlias = Union[int, float, np_int, np_float, py_td, np_td, pd_td]
r"""Type Hint for `Timedelta`."""
Time: TypeAlias = Union[DT, TD]
r"""Type Hint for `Time`."""

# Real-Time-Type-Variables
RealDTVar = TypeVar("RealDTVar", py_dt, np_dt, pd_dt)
r"""TypeVar for `Timestamp` values."""
RealTDVar = TypeVar("RealTDVar", py_td, np_td, pd_td)
r"""TypeVar for `Timedelta` values."""
RealTimeVar = TypeVar("RealTimeVar", py_dt, np_dt, pd_dt, py_td, np_td, pd_td)
r"""TypeVar for `Time` values."""


# Real-Time-Type-Hints
RealDT: TypeAlias = Union[py_dt, np_dt, pd_dt]
r"""Type Hint for real-time `Timestamp`."""
RealTD: TypeAlias = Union[py_td, np_td, pd_td]
r"""Type Hint for real-time `Timedelta`."""
RealTime: TypeAlias = Union[RealDT, RealTD]
r"""Type Hint for real-time `Time`."""


# Numpy-Time-Type-Variables
NumpyDTVar = TypeVar("NumpyDTVar", np_int, np_float, np_dt)
r"""TypeVar for `Timestamp` values."""
NumpyTDVar = TypeVar("NumpyTDVar", np_int, np_float, np_td)
r"""TypeVar for `Timedelta` values."""
NumpyTimeVar = TypeVar("NumpyTimeVar", np_int, np_float, np_dt, np_td)
r"""TypeVar for `Time` values."""
