r"""Bio-process Forecasting Task."""

__all__ = [
    # Classes
    "Kiwi_BioProcessTask",
]

from collections.abc import Callable
from functools import cached_property
from itertools import product
from typing import Any, Literal, NamedTuple, Optional

import torch
from pandas import DataFrame, Series
from torch import Tensor, jit
from torch.utils.data import DataLoader

from data.dependencies.tsdm.datasets import KIWI_RUNS
from data.dependencies.tsdm.encoders import BaseEncoder
from data.dependencies.tsdm.metrics import WRMSE
from data.dependencies.tsdm.random.samplers import HierarchicalSampler, SequenceSampler
from data.dependencies.tsdm.tasks.base import BaseTask
from data.dependencies.tsdm.utils.data import (
    MappingDataset,
    TimeSeriesDataset,
    folds_as_frame,
    folds_from_groups,
)
from data.dependencies.tsdm.utils.strings import repr_namedtuple


class Sample(NamedTuple):
    r"""A sample of the data."""

    key: tuple[tuple[int, int], slice]
    inputs: tuple[DataFrame, DataFrame]
    targets: float
    originals: Optional[tuple[DataFrame, DataFrame]] = None

    def __repr__(self) -> str:
        r"""Return string representation."""
        return repr_namedtuple(self, recursive=1)


class Kiwi_BioProcessTask(BaseTask):
    r"""A collection of bioreactor runs.

    For this task we do several simplifications

    - drop run_id 355
    - drop almost all metadata
    - restrict timepoints to start_time & end_time given in metadata.

    - timeseries for each run_id and experiment_id
    - metadata for each run_id and experiment_id

    When first do a train/test split.
    Then the goal is to learn a model in a multitask fashion on all the ts.

    To train, we sample
    1. random TS from the dataset
    2. random snippets from the sampled TS

    Questions:
    - Should each batch contain only snippets form a single TS, or is there merit to sampling
    snippets from multiple TS in each batch?

    Divide 'Glucose' by 10, 'OD600' by 20, 'DOT' by 100, 'Base' by 200, then use RMSE.
    """

    index: list[tuple[int, str]] = list(product(range(5), ("train", "test")))
    r"""Available index."""
    KeyType = tuple[Literal[0, 1, 2, 3, 4], Literal["train", "test"]]
    r"""Type Hint for Keys."""
    timeseries: DataFrame
    r"""The whole timeseries data."""
    metadata: DataFrame
    r"""The metadata."""
    observation_horizon: int = 96
    r"""The number of datapoints observed during prediction."""
    forecasting_horizon: int = 24
    r"""The number of datapoints the model should forecast."""
    preprocessor: BaseEncoder
    r"""Encoder for the observations."""
    controls: Series
    r"""The control variables."""
    targets: Series
    r"""The target variables."""
    observables: Series
    r"""The observables variables."""

    def __init__(
        self,
        *,
        forecasting_horizon: int = 24,
        observation_horizon: int = 96,
    ):
        self.forecasting_horizon = forecasting_horizon
        self.observation_horizon = observation_horizon
        self.horizon = self.observation_horizon + self.forecasting_horizon

        self.timeseries = ts = self.dataset.timeseries
        self.metadata = self.dataset.metadata
        self.units = self.dataset.units

        self.targets = targets = Series(["Base", "DOT", "Glucose", "OD600"])
        self.targets.index = self.targets.apply(ts.columns.get_loc)

        self.controls = controls = Series(
            [
                "Cumulated_feed_volume_glucose",
                "Cumulated_feed_volume_medium",
                "InducerConcentration",
                "StirringSpeed",
                "Flow_Air",
                "Temperature",
                "Probe_Volume",
            ]
        )
        controls.index = controls.apply(ts.columns.get_loc)

        self.observables = observables = Series(
            [
                "Base",
                "DOT",
                "Glucose",
                "OD600",
                "Acetate",
                "Fluo_GFP",
                "Volume",
                "pH",
            ]
        )
        observables.index = observables.apply(ts.columns.get_loc)

        assert (
            set(controls.values) | set(targets.values) | set(observables.values)
        ) == set(ts.columns)

    @cached_property
    def test_metric(self) -> Callable[..., Tensor]:
        r"""The metric to be used for evaluation."""
        ts = self.timeseries
        weights = DataFrame.from_dict(
            {
                "Base": 200,
                "DOT": 100,
                "Glucose": 10,
                "OD600": 20,
            },
            orient="index",
            columns=["inverse_weight"],
        )
        weights["col_index"] = weights.index.map(lambda x: (ts.columns == x).argmax())
        weights["weight"] = 1 / weights["inverse_weight"]
        weights["normalized"] = weights["weight"] / weights["weight"].sum()
        weights.index.name = "col"
        w = torch.tensor(weights["weight"])
        return jit.script(WRMSE(w))

    @cached_property
    def dataset(self) -> KIWI_RUNS:
        r"""Return the cached dataset."""
        dataset = KIWI_RUNS()
        dataset.metadata.drop([482], inplace=True)
        dataset.timeseries.drop([482], inplace=True)
        return dataset

    @cached_property
    def folds(self) -> DataFrame:
        r"""Return the folds."""
        md = self.dataset.metadata
        groups = md.groupby(["run_id", "color"], sort=False).ngroup()
        folds = folds_from_groups(
            groups, seed=2022, num_folds=5, train=7, valid=1, test=2
        )
        return folds_as_frame(folds)

    @cached_property
    def splits(self) -> dict[Any, tuple[DataFrame, DataFrame]]:
        r"""Return a subset of the data corresponding to the split.

        Returns
        -------
        tuple[DataFrame, DataFrame]
        """
        splits = {}
        for key in self.index:
            assert key in self.index, f"Wrong {key=}. Only {self.index} work."
            split, data_part = key

            mask = self.folds[split] == data_part
            idx = self.folds[split][mask].index
            timeseries = self.timeseries.reset_index(level=2).loc[idx]
            timeseries = timeseries.set_index("measurement_time", append=True)
            metadata = self.metadata.loc[idx]
            splits[key] = (timeseries, metadata)
        return splits

    @cached_property
    def dataloader_kwargs(self) -> dict:
        r"""Return the kwargs for the dataloader."""
        return {
            "batch_size": 1,
            "shuffle": False,
            "sampler": None,
            "batch_sampler": None,
            "num_workers": 0,
            "collate_fn": lambda *x: x,
            "pin_memory": False,
            "drop_last": False,
            "timeout": 0,
            "worker_init_fn": None,
            "prefetch_factor": 2,
            "persistent_workers": False,
        }

    def get_dataloader(
        self, key: KeyType, /, shuffle: bool = False, **dataloader_kwargs: Any
    ) -> DataLoader:
        r"""Return a dataloader for the given split.

        Parameters
        ----------
        key: KeyType,
        shuffle: bool, default False
        dataloader_kwargs: Any,

        Returns
        -------
        DataLoader
        """
        # Construct the dataset object
        ts, md = self.splits[key]
        dataset = _Dataset(
            ts,
            md,
            observables=self.observables.index,
            observation_horizon=self.observation_horizon,
            targets=self.targets.index,
        )

        TSDs = {}
        for idx in md.index:
            TSDs[idx] = TimeSeriesDataset(
                ts.loc[idx],
                metadata=md.loc[idx],
            )
        DS = MappingDataset(TSDs)

        # construct the sampler
        subsamplers = {
            key: SequenceSampler(ds, seq_len=self.horizon, stride=1, shuffle=shuffle)
            for key, ds in DS.items()
        }
        sampler = HierarchicalSampler(DS, subsamplers, shuffle=shuffle)

        # construct the dataloader
        kwargs: dict[str, Any] = {"collate_fn": lambda *x: x} | dataloader_kwargs
        return DataLoader(dataset, sampler=sampler, **kwargs)


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, ts, md, *, observables, targets, observation_horizon):
        super().__init__()
        self.timeseries = ts
        self.metadata = md
        self.observables = observables
        self.targets = targets
        self.observation_horizon = observation_horizon

    def __len__(self) -> int:
        r"""Return the number of samples in the dataset."""
        return len(self.metadata)

    def __getitem__(self, item: tuple[tuple[int, int], slice]) -> Sample:
        r"""Return a sample from the dataset."""
        key, slc = item
        ts = self.timeseries.loc[key].iloc[slc].copy(deep=True)
        md = self.metadata.loc[key].copy(deep=True)
        originals = (ts.copy(deep=True), md.copy(deep=True))
        targets = ts.iloc[self.observation_horizon :, self.targets].copy(deep=True)
        ts.iloc[self.observation_horizon :, self.targets] = float("nan")
        ts.iloc[self.observation_horizon :, self.observables] = float("nan")
        return Sample(key=item, inputs=(ts, md), targets=targets, originals=originals)
