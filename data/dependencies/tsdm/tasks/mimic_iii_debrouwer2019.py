r"""MIMIC-II clinical dataset."""

__all__ = [
    "MIMIC_III_DeBrouwer2019",
    "mimic_collate",
    "Sample",
    "Batch",
    "TaskDataset",
]

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, NamedTuple

import numpy as np
import torch
from pandas import DataFrame, Index, MultiIndex
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch import nan as NAN
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from data.dependencies.tsdm.datasets import MIMIC_III_DeBrouwer2019 as MIMIC_III_Dataset
from data.dependencies.tsdm.tasks.base import BaseTask
from data.dependencies.tsdm.utils import is_partition
from data.dependencies.tsdm.utils.strings import repr_namedtuple


class Inputs(NamedTuple):
    r"""A single sample of the data."""

    t: Tensor
    x: Tensor
    t_target: Tensor

    def __repr__(self) -> str:
        r"""Return string representation."""
        return repr_namedtuple(self, recursive=False)


class Sample(NamedTuple):
    r"""A single sample of the data."""

    key: int
    inputs: Inputs
    targets: Tensor
    originals: tuple[Tensor, Tensor]

    def __repr__(self) -> str:
        r"""Return string representation."""
        return repr_namedtuple(self, recursive=False)


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


@dataclass
class TaskDataset(Dataset):
    r"""Wrapper for creating samples of the dataset."""

    tensors: list[tuple[Tensor, Tensor]]
    observation_time: float
    prediction_steps: int
    idx_list: list

    def __len__(self) -> int:
        r"""Return the number of samples in the dataset."""
        return len(self.tensors)

    def __getitem__(self, key: int) -> Sample:
        t, x = self.tensors[key]
        observations = t <= self.observation_time
        first_target = observations.sum()
        sample_mask = slice(0, first_target)
        target_mask = slice(first_target, first_target + self.prediction_steps)
        return Sample(
            key=self.idx_list[key],
            inputs=Inputs(t[sample_mask], x[sample_mask], t[target_mask]),
            targets=x[target_mask],
            originals=(t, x),
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


# @torch.jit.script  # seems to break things
def mimic_collate(batch: list[Sample]) -> Batch:
    r"""Collate tensors into batch.

    Transform the data slightly: t, x, t_target → T, X where X[t_target:] = NAN
    """
    x_vals: list[Tensor] = []
    y_vals: list[Tensor] = []
    x_time: list[Tensor] = []
    y_time: list[Tensor] = []
    x_mask: list[Tensor] = []
    y_mask: list[Tensor] = []

    for sample in batch:
        t, x, t_target = sample.inputs
        y = sample.targets

        # get whole time interval
        time = torch.cat((t, t_target))
        sorted_idx = torch.argsort(time)

        # pad the x-values
        x_padding = torch.full(
            (t_target.shape[0], x.shape[-1]), fill_value=NAN, device=x.device
        )
        values = torch.cat((x, x_padding))

        # create a mask for looking up the target values
        mask_y = y.isfinite()
        mask_pad = torch.zeros_like(x, dtype=torch.bool)
        mask_x = torch.cat((mask_pad, mask_y))

        x_vals.append(values[sorted_idx])
        x_time.append(time[sorted_idx])
        x_mask.append(mask_x[sorted_idx])

        y_time.append(t_target)
        y_vals.append(y)
        y_mask.append(mask_y)


    x_time.append(torch.zeros(79))
    y_time.append(torch.zeros(3))
    x_vals.append(torch.zeros(79, 96))
    y_vals.append(torch.zeros(3, 96))
    x_mask.append(torch.zeros(79, 96))
    y_mask.append(torch.zeros(3, 96))

    x_time=pad_sequence(x_time, batch_first=True)
    x_vals=pad_sequence(x_vals, batch_first=True, padding_value=float("nan"))
    x_mask=pad_sequence(x_mask, batch_first=True)
    y_time=pad_sequence(y_time, batch_first=True)
    y_vals=pad_sequence(y_vals, batch_first=True, padding_value=float("nan"))
    y_mask=pad_sequence(y_mask, batch_first=True)

    x_time = x_time[:-1]
    y_time = y_time[:-1]
    x_vals = x_vals[:-1]
    y_vals = y_vals[:-1]
    x_mask = x_mask[:-1]
    y_mask = y_mask[:-1]
    
    # extra_mask = torch.rand_like(mask_train.float())
    # extra_mask = torch.where(extra_mask <= 0.2, 0, 1)
    # chance = torch.rand((1))
    # if chance.sum() <= 0.1:
    #     mask_train = extra_mask
    x = x_vals
    y = y_vals
    # x *= x_mask
    # y *= y_mask

    B, L, N = y.shape
    B, L_, N = x.shape

    shuffle=False
    idx = np.arange(N)
    if shuffle:
        np.random.shuffle(idx)

    mask_train = torch.where(x.isnan(), 0, 1)
    x_ = x.clone() #  x_ is the target value
    x_[:, -L:, :] = y 
    mask_observe = torch.where(x_.isnan(), 0, 1)
    # print((mask_observe - mask_train).sum())
    have_nan = False
    if have_nan == True:
        res = x_[:, :, idx]
    else:
        res = torch.nan_to_num(x_)[:, :, idx]


    return {
        "observed_data": res,
        "unshuffled": torch.nan_to_num(x_),
        "observed_mask": mask_observe[:, :, idx],
        "mask_train": mask_train[:, :, idx],
        "time_steps": x_time,
        "idx": idx,
        "feature_representation": torch.tensor(idx),
        'keys': [sample.key for sample in batch]
    }



class MIMIC_III_DeBrouwer2019(BaseTask):
    r"""Preprocessed subset of the MIMIC-III clinical dataset used by De Brouwer et al.

    Evaluation Protocol
    -------------------

    We use the publicly available MIMIC-III clinical database (Johnson et al., 2016), which contains
    EHR for more than 60,000 critical care patients. We select a subset of 21,250 patients with sufficient
    observations and extract 96 different longitudinal real-valued measurements over a period of 48 hours
    after patient admission. We refer the reader to Appendix K for further details on the cohort selection.
    We focus on the predictions of the next 3 measurements after a 36-hour observation window.

    The subset of 96 variables that we use in our study are shown in Table 5. For each of those, we
    harmonize the units and drop the uncertain occurrences. We also remove outliers by discarding the
    measurements outside the 5 standard deviation interval. For models requiring binning of the time
    series, we map the measurements in 30-minute time bins, which gives 97 bins for 48 hours. When
    two observations fall in the same bin, they are either averaged or summed depending on the nature
    of the observation. Using the same taxonomy as in Table 5, lab measurements are averaged, while
    inputs, outputs, and prescriptions are summed.
    This gives a total of 3,082,224 unique measurements across all patients, or an average of 145
    measurements per patient over 48 hours.

    References
    ----------
    - | `GRU-ODE-Bayes: Continuous Modeling of Sporadically-Observed Time Series
        <https://proceedings.neurips.cc/paper/2019/hash/455cb2657aaa59e32fad80cb0b65b9dc-Abstract.html>`_
      | De Brouwer, Edward and Simm, Jaak and Arany, Adam and Moreau, Yves
      | `Advances in Neural Information Processing Systems 2019
        <https://proceedings.neurips.cc/paper/2019>`_
    """

    observation_time = 72  # corresponds to 36 hours after admission (freq=30min)
    prediction_steps = 3
    num_folds = 5
    seed = 432
    test_size = 0.1  # of total
    valid_size = 0.1  # of train, i.e. 0.9*0.2 = 0.18

    def __init__(
        self, 
        normalize_time: bool = True, 
        seq_len: float = (36 - 0.5) * 2, 
        pred_len: int = 3 * 2, 
        num_folds: int = 1
    ):
        super().__init__()
        self.prediction_steps = pred_len
        self.observation_time = seq_len
        self.num_folds = num_folds
        self.normalize_time = normalize_time
        self.IDs = self.dataset.reset_index()["UNIQUE_ID"].unique()

    @cached_property
    def dataset(self) -> DataFrame:
        r"""Load the dataset."""
        ts = MIMIC_III_Dataset()["timeseries"]
        # https://github.com/edebrouwer/gru_ode_bayes/blob/aaff298c0fcc037c62050c14373ad868bffff7d2/data_preproc/Climate/generate_folds.py#L10-L14
        if self.normalize_time:
            ts = ts.reset_index()
            t_max = ts["TIME_STAMP"].max()
            self.observation_time /= t_max
            ts["TIME_STAMP"] /= t_max
            ts = ts.set_index(["UNIQUE_ID", "TIME_STAMP"])
        ts = ts.dropna(axis=1, how="all").copy()
        # catogary = np.load('clusters.npy', allow_pickle=True).item()
        # catogary = pd.DataFrame({
        #     "idx": catogary['idx'],
        #     'c': catogary['c']
        # })
        return ts
    
    def get_mean_value_according_ID(self, IDs):
        datas = []
        datas = np.array([np.nanmean(val[1], axis=0) for idx, val in self.tensors.items() if idx in IDs])
        # for id in IDs:
        #     data = self.tensors[1].loc[id]
        #     data = np.nanmean(data, axis=0)
        #     datas.append(np.expand_dims(data,0))
        # datas = np.concatenate(datas, axis=0)
        datas = np.nanmean(datas, axis=0)
        cols = list(self.dataset.columns)
        # print(cols_)
        return datas, cols


    @cached_property
    def folds(self) -> list[dict[str, Sequence[int]]]:
        r"""Create the folds."""
        num_folds = 5
        folds = []
        # https://github.com/edebrouwer/gru_ode_bayes/blob/aaff298c0fcc037c62050c14373ad868bffff7d2/data_preproc/Climate/generate_folds.py#L10-L14
        np.random.seed(self.seed)
        for _ in range(num_folds):
            train_idx, test_idx = train_test_split(self.IDs, test_size=self.test_size, shuffle=False)
            train_idx, valid_idx = train_test_split(
                train_idx, test_size=self.valid_size, shuffle=False
            )
            fold = {
                "train": train_idx,
                "val": valid_idx,
                "test": test_idx,
            }
            assert is_partition(fold.values(), union=self.IDs)
            folds.append(fold)

        return folds

    @cached_property
    def split_idx(self):
        r"""Create the split index."""
        fold_idx = Index(list(range(len(self.folds))), name="fold")
        splits = DataFrame(index=self.IDs, columns=fold_idx, dtype="string")

        for k in range(self.num_folds):
            for key, split in self.folds[k].items():
                mask = splits.index.isin(split)
                splits[k] = splits[k].where(
                    ~mask, key
                )  # where cond is false is replaces with key
        return splits

    @cached_property
    def split_idx_sparse(self) -> DataFrame:
        r"""Return sparse table with indices for each split.

        Returns
        -------
        DataFrame[bool]
        """
        df = self.split_idx
        columns = df.columns

        # get categoricals
        categories = {
            col: df[col].astype("category").dtype.categories for col in columns
        }

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

    @cached_property
    def test_metric(self) -> Callable[[Tensor, Tensor], Tensor]:
        r"""The test metric."""
        return nn.MSELoss()

    @cached_property
    def splits(self) -> Mapping:
        r"""Create the splits."""
        splits = {}
        for key in self.index:
            mask = self.split_idx_sparse[key]
            ids = self.split_idx_sparse.index[mask]
            splits[key] = self.dataset.loc[ids]
        return splits

    @cached_property
    def index(self) -> MultiIndex:
        r"""Create the index."""
        return self.split_idx_sparse.columns

    @cached_property
    def tensors(self) -> Mapping:
        r"""Tensor dictionary."""
        tensors = {}
        for _id in self.IDs:
            s = self.dataset.loc[_id]
            t = torch.tensor(s.index.values, dtype=torch.float32)
            x = torch.tensor(s.values, dtype=torch.float32)
            # c = torch.tensor(self.dataset[1][self.dataset[1]['idx'] == _id]['c'].values)
            # tensors[_id] = (t, x, c)
            tensors[_id] = (t, x)
        return tensors

    def get_dataloader(
        self, key: tuple[int, str], /, **dataloader_kwargs: Any
    ) -> DataLoader:
        r"""Return the dataloader for the given key."""
        fold, partition = key
        fold_idx = self.folds[fold][partition]
        dataset = TaskDataset(
            [val for idx, val in self.tensors.items() if idx in fold_idx],
            observation_time=self.observation_time,
            prediction_steps=self.prediction_steps,
            idx_list=fold_idx
        )
        kwargs: dict[str, Any] = {"collate_fn": mimic_collate} | dataloader_kwargs
        return DataLoader(dataset, **kwargs)

    def get_dataset(
        self,
        key: tuple[int, str]
    ) -> Dataset:
        r"""Return the dataset for the given key."""
        fold, partition = key
        fold_idx = self.folds[fold][partition]
        dataset = TaskDataset(
            [val for idx, val in self.tensors.items() if idx in fold_idx],
            observation_time=self.observation_time,
            prediction_steps=self.prediction_steps,
            idx_list=fold_idx
        )
        return dataset
