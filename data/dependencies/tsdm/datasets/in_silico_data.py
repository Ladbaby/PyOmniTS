r"""In silico experiments."""

__all__ = [
    # Classes
    "InSilicoData",
]

import shutil
from functools import cached_property
from importlib import resources
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pandas as pd

from data.dependencies.tsdm.datasets import examples
from data.dependencies.tsdm.datasets.base import SingleFrameDataset


class InSilicoData(SingleFrameDataset):
    r"""Artificially generated data, 8 runs, 7 attributes, ~465 samples.

    +---------+---------+---------+-----------+---------+-------+---------+-----------+------+
    |         | Time    | Biomass | Substrate | Acetate | DOTm  | Product | Volume    | Feed |
    +=========+=========+=========+===========+=========+=======+=========+===========+======+
    | unit    | float   | g/L     | g/l       | g/L     | %     | g/L     | L         | µL   |
    +---------+---------+---------+-----------+---------+-------+---------+-----------+------+
    | domain  | [0, 12] | >0      | >0        | >0      | [0,1] | >0      | [0, 0.01] | >0   |
    +---------+---------+---------+-----------+---------+-------+---------+-----------+------+
    | missing | -       | 99%     | 99%       | 99%     | 12%   | 99%     | 93%       | -    |
    +---------+---------+---------+-----------+---------+-------+---------+-----------+------+
    """

    DATASET_SHA256 = "f6938b4e9de35824c24c3bdc7f08c4d9bfcf9272eaeb76f579d823ca8628bff0"
    DATASET_SHAPE = (5206, 7)
    RAWDATA_SHA256 = "ee9ad6278fb27dd933c22aecfc7b5b2501336e859a7f012cace2bb265f713cba"
    rawdata_files = "in_silico.zip"

    @cached_property
    def rawdata_paths(self) -> Path:
        r"""Path to the raw data files."""
        with resources.path(examples, "in_silico.zip") as path:
            return path

    def _clean(self) -> None:
        with ZipFile(self.rawdata_paths) as files:
            dfs = {}
            for fname in files.namelist():
                key = int(fname.split(".csv")[0])
                with files.open(fname) as file:
                    df = pd.read_csv(file, index_col=0, parse_dates=[0])
                df = df.rename_axis(index="time")
                df["DOTm"] /= 100
                df.name = key
                dfs[key] = df
        ds = pd.concat(dfs, names=["run_id"])
        ds = ds.reset_index()
        ds = ds.astype({"run_id": "string"}).astype({"run_id": "category"})
        ds = ds.set_index(["run_id", "time"])
        ds = ds.sort_values(by=["run_id", "time"])
        ds = ds.astype("Float32")
        return ds

    def _download(self, **kwargs: Any) -> None:
        r"""Download the dataset."""
        with resources.path(examples, "examples/in_silico.zip") as path:
            shutil.copy(path, self.RAWDATA_DIR)
