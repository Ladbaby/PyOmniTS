r"""Code for the electricty task from the paper "Temporal Fusion Transformer" by Lim et al. (2021)."""

__all__ = [
    # CLASSES
    "ElectricityLim2021",
]


from collections.abc import Callable, Mapping, Sequence
from functools import cached_property
from typing import Any, Literal, NamedTuple, Optional

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from data.dependencies.tsdm.datasets import Electricity
from data.dependencies.tsdm.encoders import BaseEncoder, Standardizer
from data.dependencies.tsdm.random.samplers import SequenceSampler
from data.dependencies.tsdm.tasks.base import BaseTask
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


class Batch(NamedTuple):
    r"""A single sample of the data."""

    x_time: Tensor  # B×N:   the input timestamps.
    x_vals: Tensor  # B×N×D: the input values.
    x_mask: Tensor  # B×N×D: the input mask.

    y_time: Tensor  # B×K:   the target timestamps.
    y_vals: Tensor  # B×K×D: the target values.
    y_mask: Tensor  # B×K×D: teh target mask.

    def __repr__(self) -> str:
        return repr_namedtuple(self, recursive=False)


class ElectricityLim2021(BaseTask):
    r"""Experiments as performed by the "TFT" paper.

    Note that there is an issue: in the pipe-line, the hourly aggregation is done via mean,
    whereas in the TRMF paper, the hourly aggregation is done via sum.

    > We convert the data to reflect hourly consumption, by aggregating blocks of 4 columns,

    Issues:

    - They report in the paper: 90% train, 10% validation. However, this is wrong.
      They split the array not on the % of samples, but instead they use the first 218 days
      as train and the following 23 days ("2014-08-08" ≤ t < "2014-09-01" ) as validation,
      leading to a split of 90.12% train and 9.88% validation.
    - preprocessing: what partitions of the dataset are mean and variance computed over?
      train? train+validation?
    - Since the values are in kW, an hourly value would correspond to summing the values.
    - Testing: How exactly is the loss computed? From the description it can not be
      precisely inferred. Looking inside the code reveals that the test split is actually the
      last 14 days, which makes sense given that the observation period is 7 days. However,
      the paper does not mention the stride. Given the description
      "we use the past week (i.e. 168 hours) to forecast over the next 24 hours."
      Does not tell the stride, i.e. how much the sliding window is moved. We assume this to be
      24h.
    - Is the loss computed on the original scale, or on the pre-processed (i.e. z-score normalized)
      scale? The code reveals that apparently the loss is computed on the original scale!

    Paper
    -----
    - | Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
      | https://www.sciencedirect.com/science/article/pii/S0169207021000637

    Evaluation Protocol
    -------------------
    .. epigraph::

        In accordance with [9], we use the past week (i.e. 168 hours) to
        forecast over the next 24 hours.

        Electricity: Per [9], we use 500k samples taken between 2014-01-01 to 2014-09-01 – using
        the first 90% for training, and the last 10% as a validation set. Testing is done over the
        7 days immediately following the training set – as described in [9, 32]. Given the large
        differences in magnitude between trajectories, we also apply z-score normalization
        separately to each entity for real-valued inputs. In line with previous work, we consider
        the electricity usage, day-of-week, hour-of-day and a time index – i.e. the number of
        time steps from the first observation – as real-valued inputs, and treat the entity
        identifier as a categorical variable.

    Test-Metric
    -----------
    Evaluation: $q$-Risk ($q=50$ and $q=90$)

    .. math:: q-Risk = 2\frac{∑_{y_t} ∑_{τ} QL(y(t), ŷ(t-τ), q)}{∑_y ∑_{τ} |y(t)|}

    Results
    -------
    +-------+-------+-----------+-------+--------+-------+-------+---------+-------+-------+
    | Model | ARIMA | ConvTrans | DSSM  | DeepAR | ETS   | MQRNN | Seq2Seq | TFT   | TRMF  |
    +=======+=======+===========+=======+========+=======+=======+=========+=======+=======+
    | P50   | 0.154 | 0.059     | 0.083 | 0.075  | 0.102 | 0.077 | 0.067   | 0.055 | 0.084 |
    +-------+-------+-----------+-------+--------+-------+-------+---------+-------+-------+
    | P90   | 0.102 | 0.034     | 0.056 | 0.400  | 0.077 | 0.036 | 0.036   | 0.027 | NaN   |
    +-------+-------+-----------+-------+--------+-------+-------+---------+-------+-------+
    """

    KeyType = Literal["train", "test", "valid", "joint", "whole"]
    r"""Type Hint for index."""
    encoder: BaseEncoder

    def __init__(self):
        super().__init__()
        self.observation_period = pd.Timedelta("7d")
        self.forecasting_period = pd.Timedelta("1d")
        self.observation_horizon = 24 * 7
        self.forecasting_horizon = 24

        self.encoder = Standardizer()
        self.encoder.fit(self.splits["train"])

    @cached_property
    def boundaries(self) -> dict[str, pd.Timestamp]:
        r"""Start and end dates of the training, validation and test sets."""
        return {
            "start": pd.Timestamp("2014-01-01"),
            "train": pd.Timestamp("2014-08-08"),
            "valid": pd.Timestamp("2014-09-01"),
            "final": pd.Timestamp("2014-09-08"),
        }

    @cached_property
    def test_metric(self) -> Callable[..., Tensor]:
        r"""Test metric."""
        raise NotImplementedError

    @cached_property
    def dataset(self) -> pd.DataFrame:
        r"""Return the cached dataset."""
        ds = Electricity().dataset
        ds = ds.resample("1h").mean()
        mask = (self.boundaries["start"] <= ds.index) & (
            ds.index < self.boundaries["final"]
        )
        return ds[mask]

    @cached_property
    def index(self) -> Sequence[KeyType]:
        r"""List of entity identifiers."""
        return ["train", "test", "valid", "joint", "whole"]

    @cached_property
    def masks(self) -> dict[KeyType, np.ndarray]:
        r"""Masks for the training, validation and test sets."""
        return {
            "train": (self.boundaries["start"] <= self.dataset.index)
            & (self.dataset.index < self.boundaries["train"]),
            "valid": (
                self.boundaries["train"] - self.observation_period <= self.dataset.index
            )
            & (self.dataset.index < self.boundaries["valid"]),
            "test": (
                self.boundaries["valid"] - self.observation_period <= self.dataset.index
            )
            & (self.dataset.index < self.boundaries["final"]),
            "whole": (self.boundaries["start"] <= self.dataset.index)
            & (self.dataset.index < self.boundaries["final"]),
            "joint": (self.boundaries["start"] <= self.dataset.index)
            & (self.dataset.index < self.boundaries["valid"]),
        }

    @cached_property
    def splits(self) -> Mapping[KeyType, Any]:
        r"""Return cached splits of the dataset."""
        # We intentionally use these mask instead of the simpler
        # ds[lower:upper], in order to get the boundary inclusion right.

        return {
            "train": self.dataset[self.masks["train"]],
            "valid": self.dataset[self.masks["valid"]],
            "test": self.dataset[self.masks["test"]],
            "joint": self.dataset[self.masks["joint"]],
            "whole": self.dataset[self.masks["whole"]],
        }

    # @cached_property
    # def dataloader_kwargs(self) -> dict:
    #     r"""Return the kwargs for the dataloader."""
    #     return {
    #         "batch_size": 1,
    #         "shuffle": False,
    #         "sampler": None,
    #         "batch_sampler": None,
    #         "num_workers": 0,
    #         "collate_fn": lambda *x: x,
    #         "pin_memory": False,
    #         "drop_last": False,
    #         "timeout": 0,
    #         "worker_init_fn": None,
    #         "prefetch_factor": 2,
    #         "persistent_workers": False,
    #     }

    def get_dataloader(
        self, key: KeyType, /, shuffle: bool = False, **dataloader_kwargs: Any
    ) -> DataLoader:
        r"""Return the dataloader for the given key."""
        ds = self.splits[key]
        encoded = self.encoder.encode(ds)
        tensor = torch.tensor(encoded.values, dtype=torch.float32)

        sampler = SequenceSampler(
            encoded.index,
            stride="1d",
            seq_len=self.observation_period + self.forecasting_period,
            return_mask=True,
            shuffle=shuffle,
        )
        dataset = TensorDataset(tensor)

        return DataLoader(dataset, sampler=sampler, **dataloader_kwargs)
