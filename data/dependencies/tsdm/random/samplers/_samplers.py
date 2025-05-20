r"""Random Samplers."""

__all__ = [
    # ABCs
    "BaseSampler",
    "BaseSamplerMetaClass",
    # Classes
    "SliceSampler",
    # "TimeSliceSampler",
    "SequenceSampler",
    "CollectionSampler",
    "IntervalSampler",
    "HierarchicalSampler",
    "SlidingWindowSampler",
    # Functions
    "compute_grid",
]

import logging
import math
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence, Sized
from datetime import timedelta as py_td
from itertools import chain, count
from typing import Any, Generic, Literal, Optional, TypeAlias, Union, cast

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from pandas import DataFrame, Index, Series, Timedelta, Timestamp
from torch.utils.data import Sampler

from data.dependencies.tsdm.utils.data.datasets import DatasetCollection
from data.dependencies.tsdm.utils.strings import repr_mapping
from data.dependencies.tsdm.utils.types import ObjectVar, ValueVar
from data.dependencies.tsdm.utils.types.protocols import Array
from data.dependencies.tsdm.utils.types.time import DTVar, NumpyDTVar, NumpyTDVar, TDVar

Boxed: TypeAlias = Union[
    Sequence[ValueVar],
    Mapping[int, ValueVar],
    Callable[[int], ValueVar],
]

Nested: TypeAlias = Union[
    ObjectVar,
    Sequence[ObjectVar],
    Mapping[int, ObjectVar],
    Callable[[int], ObjectVar],
]


def compute_grid(
    tmin: str | DTVar,
    tmax: str | DTVar,
    timedelta: str | TDVar,
    *,
    offset: Optional[str | DTVar] = None,
) -> Sequence[int]:
    r"""Compute $\{k∈ℤ ∣ tₘᵢₙ ≤ t₀+k⋅Δt ≤ tₘₐₓ\}$.

    That is, a list of all integers such that $t₀+k⋅Δ$ is in the interval $[tₘᵢₙ, tₘₐₓ]$.
    Special case: if $Δt=0$, returns $[0]$.

    Parameters
    ----------
    tmin
    tmax
    timedelta
    offset

    Returns
    -------
    list[int]
    """
    # cast strings to timestamp/timedelta
    tmin = cast(DTVar, Timestamp(tmin) if isinstance(tmin, str) else tmin)
    tmax = cast(DTVar, Timestamp(tmax) if isinstance(tmax, str) else tmax)

    td = Timedelta(timedelta) if isinstance(timedelta, str) else timedelta

    offset = cast(
        DTVar,
        tmin
        if offset is None
        else Timestamp(offset)
        if isinstance(offset, str)
        else offset,
    )

    # offset = cast(DTVar, Timestamp(offset) if isinstance(offset, str) else offset)

    # offset = tmin if offset is None else offset
    zero_dt = tmin - tmin  # generates zero variable of correct type

    assert td > zero_dt, "Assumption ∆t>0 violated!"
    assert tmin <= offset <= tmax, "Assumption: tₘᵢₙ ≤ t₀ ≤ tₘₐₓ violated!"

    kmax = math.floor((tmax - offset) / td)
    kmin = math.ceil((tmin - offset) / td)

    return cast(Sequence[int], np.arange(kmin, kmax + 1))


class BaseSamplerMetaClass(ABCMeta):
    r"""Metaclass for BaseSampler."""

    def __init__(cls, *args, **kwargs):
        cls.LOGGER = logging.getLogger(f"{cls.__module__}.{cls.__name__}")
        super().__init__(*args, **kwargs)


class BaseSampler(Sampler, Sized, ABC, metaclass=BaseSamplerMetaClass):
    r"""Abstract Base Class for all Samplers."""

    LOGGER: logging.Logger
    r"""Logger for the sampler."""

    data: Sized
    r"""Copy of the original Data source."""

    def __init__(self, data_source: Sized, /) -> None:
        r"""Initialize the sampler."""
        super().__init__(data_source)
        self.data = data_source

    @abstractmethod
    def __len__(self) -> int:
        r"""Return the length of the sampler."""

    @abstractmethod
    def __iter__(self) -> Iterator:
        r"""Iterate over random indices."""


class SliceSampler(Sampler):
    r"""Sample by index.

    Default modus operandi:

    - Use fixed window size
    - Sample starting index uniformly from [0:-window]

    Should you want to sample windows of varying size, you may supply a

    Alternatives:

    - sample with fixed horizon and start/stop between bounds
      - [sₖ, tₖ], sᵢ = t₀ + k⋅Δt, tᵢ = t₀ + (k+1)⋅Δt
    - sample with a fixed start location and varying length.
      - [sₖ, tₖ], sᵢ = t₀, tᵢ= t₀ + k⋅Δt
    - sample with a fixed final location and varying length.
      - [sₖ, tₖ], sᵢ = tₗ - k⋅Δt, tᵢ= tₗ
    - sample with varying start and final location and varying length.
      - all slices of length k⋅Δt such that 0 < k⋅Δt < max_length
      - start stop location within bounds [t_min, t_max]
      - start stop locations from the set t_offset + [t_min, t_max] ∩ Δtℤ
      - [sₖ, tⱼ], sᵢ = t₀ + k⋅Δt, tⱼ = t₀ + k⋅Δt

    Attributes
    ----------
    data:
    idx: range(len(data))
    rng: a numpy random Generator
    """

    data: Sequence
    idx: NDArray
    rng: np.random.Generator

    def __init__(
        self,
        data_source: Sequence,
        /,
        *,
        slice_sampler: Optional[int | Callable[[], int]] = None,
        sampler: Optional[Callable[[], tuple[int, int]]] = None,
        generator: Optional[np.random.Generator] = None,
    ):
        super().__init__(data_source)
        self.data = data_source
        self.idx = np.arange(len(data_source))
        self.rng = np.random.default_rng() if generator is None else generator

        def _slicesampler_dispatch() -> Callable[[], int]:
            # use default if None is provided
            if slice_sampler is None:
                return lambda: max(1, len(data_source) // 10)
            # convert int to constant function
            if callable(slice_sampler):
                return slice_sampler
            if isinstance(slice_sampler, int):
                return lambda: slice_sampler  # type: ignore[return-value]
            raise NotImplementedError("slice_sampler not compatible.")

        self._slice_sampler = _slicesampler_dispatch()

        def _default_sampler() -> tuple[int, int]:
            window_size: int = self._slice_sampler()
            start_index: int = self.rng.choice(
                self.idx[: -1 * window_size]
            )  # -1*w silences pylint.
            return window_size, start_index

        self._sampler = _default_sampler if sampler is None else sampler

    def slice_sampler(self) -> int:
        r"""Return random window size."""
        return self._slice_sampler()

    def sampler(self) -> tuple[int, int]:
        r"""Return random start_index and window_size."""
        return self._sampler()

    def __iter__(self) -> Iterator:
        r"""Yield random slice from dataset.

        Returns
        -------
        Iterator
        """
        while True:
            # sample len and index
            window_size, start_index = self.sampler()
            # return slice
            yield self.data[start_index : start_index + window_size]


class CollectionSampler(Sampler):
    r"""Samples a single random dataset from a collection of dataset.

    Optionally, we can delegate a subsampler to then sample from the randomly drawn dataset.
    """

    idx: Index
    r"""The shared index."""
    subsamplers: Mapping[Any, Sampler]
    r"""The subsamplers to sample from the collection."""
    early_stop: bool = False
    r"""Whether to stop sampling when the index is exhausted."""
    shuffle: bool = True
    r"""Whether to sample in random order."""
    sizes: Series
    r"""The sizes of the subsamplers."""
    partition: Series
    r"""Contains each key a number of times equal to the size of the subsampler."""

    def __init__(
        self,
        data_source: DatasetCollection,
        /,
        subsamplers: Mapping[Any, Sampler],
        *,
        shuffle: bool = True,
        early_stop: bool = False,
    ):
        super().__init__(data_source)
        self.data = data_source
        self.shuffle = shuffle
        self.idx = data_source.keys()
        self.subsamplers = dict(subsamplers)
        self.early_stop = early_stop
        self.sizes = Series({key: len(self.subsamplers[key]) for key in self.idx})

        if early_stop:
            partition = list(chain(*([key] * min(self.sizes) for key in self.idx)))
        else:
            partition = list(chain(*([key] * self.sizes[key] for key in self.idx)))
        self.partition = Series(partition)

    def __len__(self):
        r"""Return the maximum allowed index."""
        if self.early_stop:
            return min(self.sizes) * len(self.subsamplers)
        return sum(self.sizes)

    def __iter__(self):
        r"""Return indices of the samples.

        When `early_stop=True`, it will sample precisely `min() * len(subsamplers)` samples.
        When `early_stop=False`, it will sample all samples.
        """
        activate_iterators = {
            key: iter(sampler) for key, sampler in self.subsamplers.items()
        }
        perm = np.random.permutation(self.partition)

        for key in perm:
            # This won't raise StopIteration, because the length is matched.
            # value = yield from activate_iterators[key]
            try:
                value = next(activate_iterators[key])
            except StopIteration as E:
                raise RuntimeError(f"Iterator of {key=} exhausted prematurely.") from E
            else:
                yield key, value

    def __getitem__(self, key: Any) -> Sampler:
        r"""Return the subsampler for the given key."""
        return self.subsamplers[key]


class HierarchicalSampler(Sampler):
    r"""Samples a single random dataset from a collection of dataset.

    Optionally, we can delegate a subsampler to then sample from the randomly drawn dataset.
    """

    idx: Index
    r"""The shared index."""
    subsamplers: Mapping[Any, Sampler]
    r"""The subsamplers to sample from the collection."""
    early_stop: bool = False
    r"""Whether to stop sampling when the index is exhausted."""
    shuffle: bool = True
    r"""Whether to sample in random order."""
    sizes: Series
    r"""The sizes of the subsamplers."""
    partition: Series
    r"""Contains each key a number of times equal to the size of the subsampler."""

    def __init__(
        self,
        data_source: Mapping[Any, Any],
        /,
        subsamplers: Mapping[Any, Sampler],
        *,
        shuffle: bool = True,
        early_stop: bool = False,
    ):
        super().__init__(data_source)
        self.data = data_source
        self.idx = Index(data_source.keys())
        self.subsamplers = dict(subsamplers)
        self.sizes = Series({key: len(self.subsamplers[key]) for key in self.idx})
        self.shuffle = shuffle
        self.early_stop = early_stop

        if early_stop:
            partition = list(chain(*([key] * min(self.sizes) for key in self.idx)))
        else:
            partition = list(chain(*([key] * self.sizes[key] for key in self.idx)))
        self.partition = Series(partition)

    def __len__(self):
        r"""Return the maximum allowed index."""
        if self.early_stop:
            return min(self.sizes) * len(self.subsamplers)
        return sum(self.sizes)

    def __iter__(self):
        r"""Return indices of the samples.

        When ``early_stop=True``, it will sample precisely ``min() * len(subsamplers)`` samples.
        When ``early_stop=False``, it will sample all samples.
        """
        activate_iterators = {
            key: iter(sampler) for key, sampler in self.subsamplers.items()
        }

        if self.shuffle:
            perm = np.random.permutation(self.partition)
        else:
            perm = self.partition

        for key in perm:
            # This won't raise StopIteration, because the length is matched.
            try:
                value = next(activate_iterators[key])
            except StopIteration as E:
                raise RuntimeError(f"Iterator of {key=} exhausted prematurely.") from E
            else:
                yield key, value

    def __getitem__(self, key: Any) -> Sampler:
        r"""Return the subsampler for the given key."""
        return self.subsamplers[key]

    def __repr__(self) -> str:
        r"""Pretty print."""
        return repr_mapping(self.subsamplers)


class IntervalSampler(Sampler, Generic[TDVar]):
    r"""Returns all intervals `[a, b]`.

    The intervals must satisfy:

    - `a = t₀ + i⋅sₖ`
    - `b = t₀ + i⋅sₖ + Δtₖ`
    - `i, k ∈ ℤ`
    - `a ≥ t_min`
    - `b ≤ t_max`
    - `sₖ` is the stride corresponding to intervals of size `Δtₖ`.
    """

    offset: TDVar
    deltax: Nested[TDVar]
    stride: Nested[TDVar]
    shuffle: bool
    intervals: DataFrame

    @staticmethod
    def _get_value(obj: TDVar | Boxed[TDVar], k: int) -> TDVar:
        if callable(obj):
            return obj(k)
        if isinstance(obj, Sequence):
            return obj[k]
        if isinstance(obj, Mapping):
            return obj[k]
        # Fallback: multiple!
        return obj

    def __init__(
        self,
        *,
        xmin: TDVar,
        xmax: TDVar,
        deltax: Nested[TDVar],
        stride: Optional[Nested[TDVar]] = None,
        levels: Optional[Sequence[int]] = None,
        offset: Optional[TDVar] = None,
        shuffle: bool = True,
    ) -> None:
        super().__init__(None)

        # set stride and offset
        zero = 0 * (xmax - xmin)
        stride = zero if stride is None else stride
        offset = xmin if offset is None else offset

        # validate bounds
        assert xmin <= offset <= xmax, "Assumption: xmin≤xoffset≤xmax violated!"

        # determine delta_max
        delta_max = max(offset - xmin, xmax - offset)

        # determine levels
        if levels is None:
            if isinstance(deltax, Mapping):
                levels = [k for k in deltax.keys() if deltax[k] <= delta_max]
            elif isinstance(deltax, Sequence):
                levels = [k for k in range(len(deltax)) if deltax[k] <= delta_max]
            elif callable(deltax):
                levels = []
                for k in count():
                    dt = self._get_value(deltax, k)
                    if dt == zero:
                        continue
                    if dt > delta_max:
                        break
                    levels.append(k)
            else:
                levels = [0]
        else:
            levels = [k for k in levels if self._get_value(deltax, k) <= delta_max]

        # validate levels
        assert all(self._get_value(deltax, k) <= delta_max for k in levels)
        # compute valid intervals
        intervals: list[
            tuple[Nested[TDVar], Nested[TDVar], Nested[TDVar], Nested[TDVar]]
        ] = []

        # for each level, get all intervals
        for k in levels:
            dt = self._get_value(deltax, k)
            st = self._get_value(stride, k)
            x0 = self._get_value(offset, k)

            # get valid interval bounds, probably there is an easier way to do it...
            stride_left: Sequence[int] = compute_grid(xmin, xmax, st, offset=x0)
            stride_right: Sequence[int] = compute_grid(xmin, xmax, st, offset=x0 + dt)
            valid_strides: set[int] = set.intersection(
                set(stride_left), set(stride_right)
            )

            if not valid_strides:
                break

            intervals.extend(
                [(x0 + i * st, x0 + i * st + dt, dt, st) for i in valid_strides]
            )

        # set variables
        self.offset = offset
        self.deltax = deltax
        self.stride = stride
        self.shuffle = shuffle
        self.intervals = DataFrame(
            intervals, columns=["left", "right", "delta", "stride"]
        )

    def __iter__(self) -> Iterator[slice]:
        r"""Return an iterator over the intervals."""
        if self.shuffle:
            perm = np.random.permutation(len(self))
        else:
            perm = np.arange(len(self))

        for k in perm:
            yield slice(self.loc[k, "left"], self.loc[k, "right"])

    def __len__(self) -> int:
        r"""Length of the sampler."""
        return len(self.intervals)

    def __getattr__(self, key: str) -> Any:
        r"""Forward all other attributes to the interval frame."""
        return self.intervals.__getattr__(key)

    def __getitem__(self, key: Any) -> slice:
        r"""Return a slice from the sampler."""
        return self.intervals[key]


class SequenceSampler(BaseSampler, Generic[DTVar, TDVar]):
    r"""Samples sequences of length seq_len."""

    data_source: Array[DTVar]
    k_max: int
    return_mask: bool
    seq_len: TDVar
    shuffle: bool
    stride: TDVar
    xmax: DTVar
    xmin: DTVar
    # total_delta: TDVar

    def __init__(
        self,
        data_source: Array[DTVar],
        *,
        seq_len: str | TDVar,
        stride: str | TDVar,
        return_mask: bool = False,
        shuffle: bool = False,
        tmax: Optional[DTVar] = None,
        tmin: Optional[DTVar] = None,
    ) -> None:
        super().__init__(data_source)
        self.data_source = data_source

        self.xmin = (
            data_source[0]
            if tmin is None
            else (Timestamp(tmin) if isinstance(tmin, str) else tmin)
        )
        self.xmax = (
            data_source[-1]
            if tmax is None
            else (Timestamp(tmax) if isinstance(tmax, str) else tmax)
        )

        total_delta = cast(TDVar, self.xmax - self.xmin)  # type: ignore[redundant-cast]
        self.stride = cast(
            TDVar, Timedelta(stride) if isinstance(stride, str) else stride
        )
        self.seq_len = cast(
            TDVar, Timedelta(seq_len) if isinstance(seq_len, str) else seq_len
        )

        # k_max = max {k∈ℕ ∣ x_min + seq_len + k⋅stride ≤ x_max}
        self.k_max = int((total_delta - self.seq_len) // self.stride)
        self.return_mask = return_mask
        self.shuffle = shuffle

        self.samples = np.array(
            [
                (x <= self.data_source) & (self.data_source < y)  # type: ignore[operator]
                if self.return_mask
                else [x, y]
                for x, y in self._iter_tuples()
            ]
        )

    def _iter_tuples(self) -> Iterator[tuple[DTVar, DTVar]]:
        x = self.xmin
        y = cast(DTVar, x + self.seq_len)  # type: ignore[operator, call-overload, redundant-cast]
        x, y = min(x, y), max(x, y)  # allows nice handling of negative seq_len
        yield x, y

        for _ in range(len(self)):
            x = x + self.stride  # type: ignore[assignment, operator, call-overload]
            y = y + self.stride  # type: ignore[assignment, operator, call-overload]
            yield x, y

    def __len__(self) -> int:
        r"""Return the number of samples."""
        return self.k_max

    def __iter__(self) -> Iterator:
        r"""Return an iterator over the samples."""
        if self.shuffle:
            perm = np.random.permutation(len(self))
        else:
            perm = np.arange(len(self))

        return iter(self.samples[perm])

    def __repr__(self) -> str:
        r"""Return a string representation of the object."""
        return f"{self.__class__.__name__}[{self.stride}, {self.seq_len}]"


class SlidingWindowSampler(BaseSampler, Generic[NumpyDTVar, NumpyTDVar]):
    r"""Sampler that generates sliding windows over an interval.

    The `SlidingWindowSampler` generates tuples.

    Inputs:

    - Ordered timestamps T
    - Starting time t_0
    - Final time t_f
    - stride ∆t (how much the sampler advances at each step) default, depending on data type of T:
        - integer: GCD(∆T)
        - float: max(⌊AVG(∆T)⌋, ε)
        - timestamp: resolution dependent.
    - horizons: TimeDelta or Tuple[TimeDelta]

    The sampler will return tuples of ``len(horizons)+1``.
    """

    data: NDArray[NumpyDTVar]
    grid: NDArray[np.integer]
    horizons: NumpyTDVar | NDArray[NumpyTDVar]
    mode: Literal["masks", "slices", "points"]
    shuffle: bool
    start_values: NDArray[NumpyDTVar]
    stride: NumpyTDVar
    tmax: NumpyDTVar
    tmin: NumpyDTVar
    offset: NumpyDTVar
    total_horizon: NumpyTDVar
    zero_td: NumpyTDVar
    multi_horizon: bool
    cumulative_horizons: NDArray[NumpyTDVar]

    def __init__(
        self,
        data_source: Sequence[NumpyDTVar],
        /,
        *,
        stride: str | NumpyTDVar,
        horizons: str | Sequence[str] | NumpyTDVar | Sequence[NumpyTDVar],
        tmin: Optional[str | NumpyDTVar] = None,
        tmax: Optional[str | NumpyDTVar] = None,
        mode: Literal["masks", "slices", "points"] = "masks",
        shuffle: bool = False,
    ):
        super().__init__(data_source)

        # coerce non-numpy types to numpy.
        horizons = Timedelta(horizons) if isinstance(horizons, str) else horizons
        stride = Timedelta(stride) if isinstance(stride, str) else stride
        tmin = Timestamp(tmin) if isinstance(tmin, str) else tmin
        tmax = Timestamp(tmax) if isinstance(tmax, str) else tmax

        self.shuffle = shuffle
        self.mode = mode
        self.stride = stride

        if tmin is None:
            if isinstance(self.data, (Series, DataFrame)):
                self.tmin = self.data.iloc[0]
            else:
                self.tmin = self.data[0]
        else:
            self.tmin = tmin

        if tmax is None:
            if isinstance(self.data, (Series, DataFrame)):
                self.tmax = self.data.iloc[-1]
            else:
                self.tmax = self.data[-1]
        else:
            self.tmax = tmax

        # this gives us the correct zero, depending on the dtype
        self.zero_td = self.tmin - self.tmin  # type: ignore[assignment]
        assert self.stride > self.zero_td, "stride cannot be zero."

        if isinstance(horizons, Sequence):
            self.multi_horizon = True
            if isinstance(horizons[0], (str, Timedelta, py_td)):
                self.horizons = pd.to_timedelta(horizons)
                concat_horizons = self.horizons.insert(0, self.zero_td)  # type: ignore[union-attr]
            else:
                self.horizons = np.array(horizons)
                concat_horizons = np.concatenate(([self.zero_td], self.horizons))  # type: ignore[arg-type]

            self.cumulative_horizons = np.cumsum(concat_horizons)
            self.total_horizon = self.cumulative_horizons[-1]
        else:
            self.multi_horizon = False
            self.horizons = horizons
            self.total_horizon = self.horizons
            self.cumulative_horizons = np.cumsum([self.zero_td, self.horizons])

        self.start_values = self.tmin + self.cumulative_horizons  # type: ignore[assignment, call-overload, operator]

        self.offset = self.tmin + self.total_horizon  # type: ignore[assignment, call-overload, operator]

        # precompute the possible slices
        grid = compute_grid(self.tmin, self.tmax, self.stride, offset=self.offset)
        self.grid = grid[grid >= 0]  # type: ignore[assignment, operator]

    def __len__(self):
        r"""Return the number of samples."""
        return len(self.data)

    @staticmethod
    def __make__points__(bounds: NDArray[NumpyDTVar]) -> NDArray[NumpyDTVar]:
        r"""Return the points as-is."""
        return bounds

    @staticmethod
    def __make__slice__(window: NDArray[NumpyDTVar]) -> slice:
        r"""Return a tuple of slices."""
        return slice(window[0], window[-1])

    @staticmethod
    def __make__slices__(bounds: NDArray[NumpyDTVar]) -> tuple[slice, ...]:
        r"""Return a tuple of slices."""
        return tuple(
            slice(start, stop) for start, stop in sliding_window_view(bounds, 2)
        )

    def __make__mask__(self, window: NDArray[NumpyDTVar]) -> NDArray[np.bool_]:
        r"""Return a tuple of masks."""
        return (window[0] <= self.data) & (self.data < window[-1])

    def __make__masks__(
        self, bounds: NDArray[NumpyDTVar]
    ) -> tuple[NDArray[np.bool_], ...]:
        r"""Return a tuple of masks."""
        return tuple(
            (start <= self.data) & (self.data < stop)
            for start, stop in sliding_window_view(bounds, 2)
        )

    def __iter__(self) -> Iterator:
        r"""Iterate through.

        For each k, we return either:

        - mode=points: $(x₀ + k⋅∆t, x₁+k⋅∆t, …, xₘ+k⋅∆t)$
        - mode=slices: $(slice(x₀ + k⋅∆t, x₁+k⋅∆t), …, slice(xₘ₋₁+k⋅∆t, xₘ+k⋅∆t))$
        - mode=masks: $(mask_1, …, mask_m)$
        """
        yield_fn: Callable[[NDArray[NumpyDTVar]], Any]
        if self.mode == "points":
            yield_fn = self.__make__points__
        else:
            yield_fn = {
                ("masks", False): self.__make__mask__,
                ("masks", True): self.__make__masks__,
                ("slices", False): self.__make__slice__,
                ("slices", True): self.__make__slices__,
            }[(self.mode, self.multi_horizon)]

        if self.shuffle:
            perm = np.random.permutation(len(self.grid))
            grid = self.grid[perm]
        else:
            grid = self.grid

        for k in grid:
            vals = self.start_values + k * self.stride
            yield yield_fn(vals)
