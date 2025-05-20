r"""Electricity Transformer Dataset (ETDataset).

**Source:** https://github.com/zhouhaoyi/ETDataset
"""

__all__ = [
    # Classes
    "ETT"
]

from pathlib import Path
from typing import Literal

from pandas import read_csv

from data.dependencies.tsdm.datasets.base import MultiFrameDataset


class ETT(MultiFrameDataset):
    r"""ETT-small-h1.

    +-------------+-------------------+------------------+-------------------+--------------------+---------------------+-----------------+------------------+--------------------------+
    | Field       | date              | HUFL             | HULL              | MUFL               | MULL                | LUFL            | LULL             | OT                       |
    +=============+===================+==================+===================+====================+=====================+=================+==================+==========================+
    | Description | The recorded date | High UseFul Load | High UseLess Load | Middle UseFul Load | Middle UseLess Load | Low UseFul Load | Low UseLess Load | Oil Temperature (target) |
    +-------------+-------------------+------------------+-------------------+--------------------+---------------------+-----------------+------------------+--------------------------+
    """  # pylint: disable=line-too-long # noqa: E501

    BASE_URL = r"https://github.com/zhouhaoyi/ETDataset/tree/main/ETT-small/"
    r"""HTTP address from where the dataset can be downloaded."""
    INFO_URL = r"https://github.com/zhouhaoyi/ETDataset"
    r"""HTTP address containing additional information about the dataset."""
    DATASET_SHA256 = {
        "ETTh1": "b56abe3a5a0ac54428be73a37249d549440a7512fce182adcafba9ee43a03694",
        "ETTh2": "0607d0f59341e87f2ab0f520fb885ad6983aa5b17b058fc802ebd87c51f75387",
        "ETTm1": "62df6ea49e60b9e43e105b694e539e572ba1d06bda4df283faf53760d8cbd5c1",
        "ETTm2": "3c946e0fefc5c1a440e7842cdfeb7f6372a1b61b3da51519d0fb4ab8eb9debad",
    }
    RAWDATA_SHA256 = {
        "ETTh1.csv": "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066",
        "ETTh2.csv": "a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b",
        "ETTm1.csv": "6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e",
        "ETTm2.csv": "db973ca252c6410a30d0469b13d696cf919648d0f3fd588c60f03fdbdbadd1fd",
    }
    DATASET_SHAPE = {
        "ETTh1.csv": (17420, 7),
        "ETTh2.csv": (17420, 7),
        "ETTm1.csv": (69680, 7),
        "ETTm2.csv": (69680, 7),
    }
    KEYS = Literal["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
    r"""The type of the index."""
    index: list[KEYS] = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
    r"""IDs of the stored data-objects."""
    rawdata_files = {key: f"{key}.csv" for key in index}
    r"""Files containing the raw data."""
    rawdata_paths: dict[KEYS, Path]
    r"""Paths to the raw data."""

    def _clean(self, key: KEYS) -> None:
        r"""Create DataFrame from the .csv file."""
        df = read_csv(
            self.rawdata_paths[key], parse_dates=[0], index_col=0, dtype="float32"
        )
        return df
