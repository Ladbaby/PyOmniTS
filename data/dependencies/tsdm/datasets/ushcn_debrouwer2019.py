r"""#TODO add module summary line.

#TODO add module description.
"""

__all__ = [
    # Classes
    "USHCN_DeBrouwer2019",
]

import logging
from pathlib import Path

import pandas
from pandas import DataFrame

from data.dependencies.tsdm.datasets.base import SingleFrameDataset

__logger__ = logging.getLogger(__name__)


class USHCN_DeBrouwer2019(SingleFrameDataset):
    r"""Preprocessed subset of the USHCN climate dataset used by De Brouwer et. al.

    References
    ----------
    - | `GRU-ODE-Bayes: Continuous Modeling of Sporadically-Observed Time Series
        <https://proceedings.neurips.cc/paper/2019/hash/455cb2657aaa59e32fad80cb0b65b9dc-Abstract.html>`_
      | De Brouwer, Edward and Simm, Jaak and Arany, Adam and Moreau, Yves
      | `Advances in Neural Information Processing Systems 2019
        <https://proceedings.neurips.cc/paper/2019>`_
    """

    BASE_URL = (
        r"https://raw.githubusercontent.com/edebrouwer/gru_ode_bayes/"
        r"master/gru_ode_bayes/datasets/Climate/"
    )
    r"""HTTP address from where the dataset can be downloaded."""

    INFO_URL = "https://github.com/edebrouwer/gru_ode_bayes"
    r"""HTTP address containing additional information about the dataset."""
    DATASET_SHA256 = "bbd12ab38b4b7f9c69a07409c26967fe16af3b608daae9816312859199b5ce86"
    DATASET_SHAPE = (350665, 5)
    RAWDATA_SHA256 = "671eb8d121522e98891c84197742a6c9e9bb5015e42b328a93ebdf2cfd393ecf"
    RAWDATA_SHAPE = (350665, 12)

    rawdata_files = "small_chunked_sporadic.csv"
    rawdata_paths: Path
    # dataset_files = "SmallChunkedSporadic.feather"

    def _clean(self) -> None:
        r"""Clean an already downloaded raw dataset and stores it in hdf5 format."""
        dtypes = {
            "ID": "int16",
            "Time": "float32",
            "Value_0": "float32",
            "Value_1": "float32",
            "Value_2": "float32",
            "Value_3": "float32",
            "Value_4": "float32",
            "Mask_0": "bool",
            "Mask_1": "bool",
            "Mask_2": "bool",
            "Mask_3": "bool",
            "Mask_4": "bool",
        }
        df = pandas.read_csv(self.rawdata_paths, dtype=dtypes)
        df = DataFrame(df)

        if df.shape != self.RAWDATA_SHAPE:
            raise ValueError(
                f"The {df.shape=} is not correct."
                "Please apply the modified preprocessing using bin_k=2, as outlined in"
                "the appendix. The resulting tensor should have 3082224 rows and 7 columns."
            )

        channels = {}
        for k in range(5):
            key = f"CH_{k}"
            value = f"Value_{k}"
            channels[key] = value
            df[key] = df[value].where(df[f"Mask_{k}"])

        df = df[["ID", "Time", *channels]]
        df = df.sort_values(["ID", "Time"])
        df = df.set_index(["ID", "Time"])
        df = df.rename(columns=channels)
        # df = df.reset_index(drop=True)
        return df
