r"""KIWI Run Data.

Extracted from iLab DataBase.
"""

__all__ = [
    # Classes
    "KIWI_RUNS",
]

import logging
import pickle
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final, Literal, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from data.dependencies.tsdm.datasets.base import MultiFrameDataset
from data.dependencies.tsdm.utils import round_relative

__logger__ = logging.getLogger(__name__)


def contains_no_information(series: Series) -> bool:
    r"""Check if a series contains no information."""
    return len(series.dropna().unique()) <= 1


def contains_nan_slice(
    series: Series, slices: list[Sequence], two_enough: bool = False
) -> bool:
    r"""Check if data is completely missing for many slices."""
    num_missing = 0
    for idx in slices:
        if pd.isna(series[idx]).all():
            num_missing += 1

    if (num_missing > 0 and not two_enough) or (
        num_missing >= len(slices) - 1 and two_enough
    ):
        __logger__.debug(
            "%s: data missing in %s/%s slices!", series.name, num_missing, len(slices)
        )
        return True
    return False


def float_is_int(series: Series) -> bool:
    r"""Check if all float values are integers."""
    mask = pd.notna(series)
    return series[mask].apply(float.is_integer).all()


def get_integer_cols(table: DataFrame) -> set[str]:
    r"""Get all columns that contain only integers."""
    cols = set()
    for col in table:
        if np.issubdtype(table[col].dtype, np.integer):
            __logger__.debug("Integer column                       : %s", col)
            cols.add(col)
        elif np.issubdtype(table[col].dtype, np.floating) and float_is_int(table[col]):
            __logger__.debug("Integer column pretending to be float: %s", col)
            cols.add(col)
    return cols


def get_useless_cols(
    table: DataFrame, slices: Optional[list[Sequence]] = None, strict: bool = False
) -> set[str]:
    r"""Get all columns that are considered useless."""
    useless_cols = set()
    for col in table:
        s = table[col]
        if col in ("run_id", "experiment_id"):
            continue
        if contains_no_information(s):
            __logger__.debug("No information in      %s", col)
            useless_cols.add(col)
        elif slices is not None and contains_nan_slice(
            s, slices, two_enough=(not strict)
        ):
            __logger__.debug("Missing for some run   %s", col)
            useless_cols.add(col)
    return useless_cols


class KIWI_RUNS(MultiFrameDataset):
    r"""KIWI RUN Data.

    The cleaned data will consist of 2 parts:

    - timeseries
    - metadata

    Rawdata Format:

    .. code-block:: python

        dict[int, # run_id
            dict[int, # experiment_id
                 dict[
                     'metadata',: DataFrame,                # static
                     'setpoints': DataFrame,                # static
                     'measurements_reactor',: DataFrame,    # TimeTensor
                     'measurements_array',: DataFrame,      # TimeTensor
                     'measurements_aggregated': DataFrame,  # TimeTensor
                 ]
            ]
        ]
    """

    BASE_URL = (
        "https://owncloud.innocampus.tu-berlin.de/index.php/s/fGFEJicrcjsxDBd/download/"
    )
    RAWDATA_SHA256 = "79d8d15069b4adc6d77498472008bd87e3699a75bb612029232bd051ecdbb078"
    DATASET_SHA256 = {
        "timeseries": "819d5917c5ed65cec7855f02156db1abb81ca3286e57533ee15eb91c072323f9",
        "metadata": "8b4d3f922c2fb3988ae606021492aa10dd3d420b3c6270027f91660a909429ae",
        "units": "aa4d0dd22e0e44c78e7034eb49ed39cde371fa1e4bf9b9276e9e2941c54e5eca",
    }
    DATASET_SHAPE = {
        "timeseries": (386812, 15),
        "metadata": (264, 11),
        "units": (15, 11),
    }

    index: Final[list[str]] = [
        "timeseries",
        "metadata",
        "units",
    ]
    r"""Available index."""

    auxiliaries: Final[list[str]] = [
        # "setpoints",
        # "measurements_reactor",
        # "measurements_array",
        "measurements_aggregated",
    ]

    KEYS = Literal[
        "metadata",
        # "setpoints",
        # "measurements_reactor",
        # "measurements_array",
        "measurements_aggregated",
        "timeseries",
        "units",
    ]
    r"""Type Hint for index."""

    timeseries: DataFrame
    r"""The whole timeseries data."""
    metadata: DataFrame
    r"""The metadata."""
    units: DataFrame
    r"""The units of the measured variables."""
    rawdata_files = "kiwi_experiments.pk"
    rawdata_paths: Path
    dataset_files = {key: f"{key}.parquet" for key in index + auxiliaries}

    # def _load(self, key: KEYS = "timeseries") -> DataFrame:
    #     r"""Load the dataset from disk."""
    #     table = pd.read_feather(self.dataset_paths[key])
    #
    #     if key == "units":
    #         return table.set_index("variable")
    #
    #     # fix index dtype (groupby messes it up....)
    #     table = table.astype({"run_id": "int32", "experiment_id": "int32"})
    #     if "measurements" in key or key == "timeseries":
    #         table = table.set_index(["run_id", "experiment_id", "measurement_time"])
    #         table.columns.name = "variable"
    #     else:
    #         table = table.set_index(["run_id", "experiment_id"])
    #
    #     return table

    def _clean(self, key: KEYS) -> None:
        r"""Clean an already downloaded raw dataset and stores it in feather format."""
        with open(self.rawdata_paths, "rb") as file:
            self.LOGGER.info("Loading raw data from %s", self.rawdata_paths)
            data = pickle.load(file)

        DATA = [
            (data[run][exp] | {"run_id": run, "experiment_id": exp})
            for run in data
            for exp in data[run]
        ]
        DF = DataFrame(DATA).set_index(["run_id", "experiment_id"])

        tables: dict[Any, DataFrame] = {}

        # must clean auxiliaries first
        # for key in self.auxiliaries+self.index:
        if key in ("units", "timeseries"):
            self._clean_table(key)
        elif key == "metadata":
            tables[key] = pd.concat(iter(DF[key])).reset_index(drop=True)
            self._clean_table(key, tables[key])
        else:
            tables[key] = (
                pd.concat(iter(DF[key]), keys=DF[key].index)
                .reset_index(level=2, drop=True)
                .reset_index()
            )
            self._clean_table(key, tables[key])

    def _clean_table(self, key: str, table: Optional[DataFrame] = None) -> None:
        r"""Create the DataFrames.

        Parameters
        ----------
        table: Optional[DataFrame] = None
        """
        cleaners: dict[str, Callable] = {
            "measurements_aggregated": self._clean_measurements_aggregated,
            "measurements_array": self._clean_measurements_array,
            "measurements_reactor": self._clean_measurements_reactor,
            "metadata": self._clean_metadata,
            "setpoints": self._clean_setpoints,
            "timeseries": self._clean_timeseries,
            "units": self._clean_units,
        }
        cleaner = cleaners[key]
        if table is None:
            cleaner()
        else:
            cleaner(table)

        self.LOGGER.info("%s Finished cleaning table!", key)

    def _clean_metadata(self, table: DataFrame) -> None:
        runs = table["run_id"].dropna().unique()
        run_masks: list[Series] = [table["run_id"] == run for run in runs]

        table_columns = set(table.columns)
        useless_cols = get_useless_cols(table, slices=run_masks) | {
            "folder_id_y",
            "ph_Base_conc",
            "ph_Ki",
            "ph_Kp",
            "ph_Tolerance",
            "pms_id",
            "description",
        }
        get_integer_cols(table)
        remaining_cols = table_columns - useless_cols

        selected_columns = {
            "Feed_concentration_glc": "float32",
            "OD_Dilution": "float32",
            "bioreactor_id": "UInt32",
            "color": "string",
            "container_number": "UInt32",
            "end_time": "datetime64[ns]",
            "experiment_id": "UInt32",
            # "organism_id": "UInt32",
            "pH_correction_factor": "float32",
            "profile_id": "UInt32",
            "profile_name": "string",
            "run_id": "UInt32",
            "run_name": "string",
            "start_time": "datetime64[ns]",
        }

        categorical_columns = {
            "Feed_concentration_glc": "Int16",
            "OD_Dilution": "Float32",
            "color": "category",
            "pH_correction_factor": "Float32",
            "profile_name": "category",
            "run_name": "category",
        }

        assert (
            selected_columns.keys() >= remaining_cols
        ), f"Missing encoding: {remaining_cols - selected_columns.keys()}"

        assert (
            selected_columns.keys() <= remaining_cols
        ), f"Superfluous encoding: {selected_columns.keys() - remaining_cols}"

        assert set(categorical_columns) <= set(
            selected_columns
        ), f"Superfluous encoding: {set(categorical_columns) - set(selected_columns)}"

        table = table[selected_columns.keys()]
        table = table.astype(selected_columns)
        table = table.astype(categorical_columns)
        table = table.reset_index(drop=True)
        table = table.set_index(["run_id", "experiment_id"])
        # table = table.rename(columns={col: snake2camel(col) for col in table})
        table.columns.name = "variable"
        table.to_parquet(self.dataset_paths["metadata"], compression="gzip")

    def _clean_setpoints(self, table: DataFrame) -> None:
        runs = table["run_id"].dropna().unique()
        run_masks: list[Series[bool]] = [table["run_id"] == run for run in runs]

        table_columns = set(table.columns)
        useless_cols = get_useless_cols(table, slices=run_masks)
        get_integer_cols(table)
        remaining_cols = table_columns - useless_cols

        selected_columns = {
            "experiment_id": "UInt32",
            "run_id": "UInt32",
            "cultivation_age": "UInt32",
            "setpoint_id": "UInt32",
            "unit": "string",
            # "Puls_AceticAcid": "Float32",
            "Puls_Glucose": "Float32",
            # "Puls_Medium": "Float32",
            "StirringSpeed": "UInt16",
            # "pH": "Float32",
            "Feed_glc_cum_setpoints": "UInt16",
            "Flow_Air": "UInt8",
            "InducerConcentration": "Float32",
            # "Flow_Nitrogen": "Float32",
            # "Flow_O2": "Float32",
            # "Feed_dextrine_cum_setpoints": "Float32",
        }

        categorical_columns = {
            "unit": "category",
        }

        assert (
            selected_columns.keys() >= remaining_cols
        ), f"Missing encoding: {remaining_cols - selected_columns.keys()}"

        assert (
            selected_columns.keys() <= remaining_cols
        ), f"Superfluous encoding: {selected_columns.keys() - remaining_cols}"

        assert set(categorical_columns) <= set(
            selected_columns
        ), f"Superfluous encoding: {set(categorical_columns) - set(selected_columns)}"

        table["unit"] = table["unit"].replace(to_replace="-", value=pd.NA)
        table = table[selected_columns.keys()]
        table = table.astype(selected_columns)
        table = table.astype(categorical_columns)
        table = table.reset_index(drop=True)
        table = table.set_index(["run_id", "experiment_id"])
        # table = table.rename(columns={col: snake2camel(col) for col in table})
        table.to_parquet(self.dataset_paths["setpoints"], compression="gzip")

    def _clean_measurements_reactor(self, table: DataFrame) -> None:
        runs = table["run_id"].dropna().unique()
        run_masks: list[Series[bool]] = [table["run_id"] == run for run in runs]

        table_columns = set(table.columns)
        useless_cols = get_useless_cols(table, slices=run_masks)
        get_integer_cols(table)
        remaining_cols = table_columns - useless_cols

        selected_columns: dict[str, str] = {
            "Acetate": "Float32",
            "Base": "Int16",
            "Cumulated_feed_volume_glucose": "Int16",
            "Cumulated_feed_volume_medium": "Float32",
            "DOT": "Float32",
            "Fluo_GFP": "Float32",
            "Glucose": "Float32",
            "InducerConcentration": "Float32",
            "OD600": "Float32",
            "Probe_Volume": "Int16",
            "Volume": "Float32",
            "experiment_id": "UInt32",
            "measurement_id": "UInt32",
            "measurement_time": "datetime64[ns]",
            "pH": "Float32",
            "run_id": "UInt32",
            "unit": "string",
        }

        categorical_columns: dict[str, str] = {
            "unit": "category",
        }

        assert (
            selected_columns.keys() >= remaining_cols
        ), f"Missing encoding: {remaining_cols - selected_columns.keys()}"

        assert (
            selected_columns.keys() <= remaining_cols
        ), f"Superfluous encoding: {selected_columns.keys() - remaining_cols}"

        assert set(categorical_columns) <= set(
            selected_columns
        ), f"Superfluous encoding: {set(categorical_columns) - set(selected_columns)}"

        table["unit"] = table["unit"].replace(to_replace="-", value=pd.NA)
        table = table[selected_columns.keys()]
        table = table.astype(selected_columns)
        table = table.astype(categorical_columns)
        table = table.reset_index(drop=True)
        table = table.set_index(["run_id", "experiment_id", "measurement_time"])
        # table = table.rename(columns={col: snake2camel(col) for col in table})
        table.to_parquet(self.dataset_paths["measurements_reactor"], compression="gzip")

    def _clean_measurements_array(self, table: DataFrame) -> None:
        runs = table["run_id"].dropna().unique()
        run_masks: list[Series[bool]] = [table["run_id"] == run for run in runs]

        table_columns = set(table.columns)
        useless_cols = get_useless_cols(table, slices=run_masks)
        get_integer_cols(table)
        remaining_cols = table_columns - useless_cols

        selected_columns: dict[str, str] = {
            "run_id": "UInt32",
            "experiment_id": "UInt32",
            "measurement_time": "datetime64[ns]",
            "measurement_id": "UInt32",
            "unit": "string",
            "Flow_Air": "Float32",
            # "Flow_Nitrogen"      :         "float64",
            # "Flow_O2"            :         "float64",
            "StirringSpeed": "Int16",
            "Temperature": "Float32",
        }

        categorical_columns: dict[str, str] = {"unit": "category"}

        assert (
            selected_columns.keys() >= remaining_cols
        ), f"Missing encoding: {remaining_cols - selected_columns.keys()}"

        assert (
            selected_columns.keys() <= remaining_cols
        ), f"Superfluous encoding: {selected_columns.keys() - remaining_cols}"

        assert set(categorical_columns) <= set(
            selected_columns
        ), f"Superfluous encoding: {set(categorical_columns) - set(selected_columns)}"

        table["unit"] = table["unit"].replace(to_replace="-", value=pd.NA)
        table = table[selected_columns.keys()]
        table = table.astype(selected_columns)
        table = table.astype(categorical_columns)
        table = table.reset_index(drop=True)
        table = table.set_index(["run_id", "experiment_id", "measurement_time"])
        # table = table.rename(columns={col: snake2camel(col) for col in table})
        table.to_parquet(self.dataset_paths["measurements_array"], compression="gzip")

    def _clean_measurements_aggregated(self, table: DataFrame) -> None:
        runs = table["run_id"].dropna().unique()
        run_masks: list[Series[bool]] = [table["run_id"] == run for run in runs]
        table_columns = set(table.columns)
        useless_cols = get_useless_cols(table, slices=run_masks)
        get_integer_cols(table)
        remaining_cols = table_columns - useless_cols

        selected_columns: dict[str, str] = {
            "run_id": "UInt32",
            "experiment_id": "UInt32",
            "measurement_time": "datetime64[ns]",
            "unit": "string",
            "Flow_Air": "Float32",
            # "Flow_Nitrogen"                 :          "Float32",
            # "Flow_O2"                       :          "Int32",
            "StirringSpeed": "Int16",
            "Temperature": "Float32",
            "Acetate": "Float32",
            # "Acid"                          :          "Float32",
            "Base": "Int16",
            "Cumulated_feed_volume_glucose": "Int16",
            "Cumulated_feed_volume_medium": "Float32",
            "DOT": "Float32",
            # "Fluo_CFP"                      :          "Float32",
            # "Fluo_RFP"                      :          "Float32",
            # "Fluo_YFP"                      :          "Float32",
            "Glucose": "Float32",
            "OD600": "Float32",
            "Probe_Volume": "Int16",
            "pH": "Float32",
            "Fluo_GFP": "Float32",
            "InducerConcentration": "Float32",
            # "remark"                        :           "string",
            "Volume": "Float32",
        }
        categorical_columns: dict[str, str] = {"unit": "category"}

        assert (
            selected_columns.keys() >= remaining_cols
        ), f"Missing encoding: {remaining_cols - selected_columns.keys()}"

        assert (
            selected_columns.keys() <= remaining_cols
        ), f"Superfluous encoding: {selected_columns.keys() - remaining_cols}"

        assert set(categorical_columns) <= set(
            selected_columns
        ), f"Superfluous encoding: {set(categorical_columns) - set(selected_columns)}"

        table["unit"] = table["unit"].replace(to_replace="-", value=pd.NA)
        table = table[selected_columns.keys()]
        table = table.astype(selected_columns)
        table = table.astype(categorical_columns)
        table = table.reset_index(drop=True)
        table = table.set_index(["run_id", "experiment_id", "measurement_time"])
        # table = table.rename(columns={col: snake2camel(col) for col in table})
        table.to_parquet(
            self.dataset_paths["measurements_aggregated"], compression="gzip"
        )

    def _clean_timeseries(self) -> None:
        md: DataFrame = self.load(key="metadata")
        ts: DataFrame = self.load(key="measurements_aggregated")
        # drop rows with only <NA> values
        ts = ts.dropna(how="all")
        # generate timeseries frame
        ts = ts.drop(columns="unit")
        # gather
        ts = ts.groupby(["run_id", "experiment_id", "measurement_time"]).mean()
        # drop rows with only <NA> values
        ts = ts.dropna(how="all")
        # convert all value columns to float
        ts = ts.astype("Float32")
        # sort according to measurement_time
        ts = ts.sort_values(["run_id", "experiment_id", "measurement_time"])
        # drop all time stamps outside of start_time and end_time in the metadata.
        merged = ts[[]].join(md[["start_time", "end_time"]])
        time = merged.index.get_level_values("measurement_time")
        cond = (merged["start_time"] <= time) & (time <= merged["end_time"])
        ts = ts[cond]

        # Change index to TimeDelta
        realtime = merged.index.get_level_values("measurement_time") - merged.start_time
        realtime.name = "measurement_time"
        ts = ts.join(realtime)
        ts = ts.reset_index(-1, drop=True)
        ts = ts.set_index("measurement_time", append=True)

        # replace wrong pH values
        ts["pH"] = ts["pH"].replace(0.0, pd.NA)
        # mask out-of bounds values
        ts["Glucose"][ts["Glucose"] < 0] = pd.NA
        ts["Acetate"][ts["Acetate"] < 0] = pd.NA
        ts["Fluo_GFP"][ts["Fluo_GFP"] < 0] = pd.NA
        ts["Fluo_GFP"][ts["Fluo_GFP"] > 1_000_000] = 1_000_000
        ts["DOT"][ts["DOT"] < 0] = pd.NA
        ts["DOT"][ts["DOT"] > 100] = 100.0
        ts["OD600"][ts["OD600"] < 0] = pd.NA
        ts["OD600"][ts["OD600"] > 100] = 100
        ts["Volume"][ts["Volume"] < 0] = pd.NA

        # drop rows with only <NA> values
        ts = ts.dropna(how="all")

        # Forward Fill piece-wise constant control variables
        ffill_consts = [
            "Cumulated_feed_volume_glucose",
            "Cumulated_feed_volume_medium",
            "InducerConcentration",
            "StirringSpeed",
            "Flow_Air",
            "Probe_Volume",
        ]
        for idx, slc in ts.groupby(["run_id", "experiment_id"]):
            slc = slc[ffill_consts].fillna(method="ffill")
            # forward fill remaining NA with zeros.
            ts.loc[idx, ffill_consts] = slc[ffill_consts].fillna(0)

        # check if metadata-index matches with times-series index
        ts_idx = ts.reset_index(level="measurement_time").index
        pd.testing.assert_index_equal(md.index, ts_idx.unique())

        # reset index
        ts = ts.reset_index()
        ts = ts.astype({"run_id": "int32", "experiment_id": "int32"})
        ts["measurement_time"] = ts["measurement_time"].round("s")
        # ts = ts.rename(columns={col: snake2camel(col) for col in ts})
        ts.columns.name = "variable"
        ts = ts.set_index(["run_id", "experiment_id", "measurement_time"])
        ts.to_parquet(self.dataset_paths["timeseries"], compression="gzip")

    def _clean_units(self) -> None:
        ts: DataFrame = self.load(key="measurements_aggregated")

        _units = ts["unit"]
        data = ts.drop(columns="unit")
        data = data.astype("float32")

        units = Series(dtype=pd.StringDtype(), name="unit")

        for col in data:
            if col == "runtime":
                continue
            mask = pd.notna(data[col])
            unit = _units[mask].unique().to_list()
            assert len(unit) <= 1, f"{col}, {unit} {len(unit)}"
            units[col] = unit[0]

        units["DOT"] = "%"
        units["OD600"] = "%"
        units["pH"] = "-log[H⁺]"
        units["Acetate"] = "g/L"
        units = units.fillna(pd.NA).astype("string").astype("category")
        units.index.name = "variable"

        # add dataset statistics
        units = units.to_frame()
        ts = self._load(key="timeseries")

        units["min"] = ts.min()
        units["max"] = ts.max()
        units["eps"] = ts[ts > units["min"]].min()
        units["sup"] = ts[ts < units["max"]].max()
        units["mean"] = ts.mean()
        units["median"] = ts.median()
        units["std"] = ts.std()
        units["nan"] = ts.isna().mean()
        units["mins"] = (ts == units["min"]).mean()
        units["maxs"] = (ts == units["max"]).mean()

        columns = [
            "min",
            "eps",
            "sup",
            "max",
            "median",
            "mean",
            "std",
            "nan",
            "mins",
            "maxs",
        ]
        percents = [
            "nan",
            "mins",
            "maxs",
        ]

        units[columns] = units[columns].astype("float32").apply(round_relative)
        units[percents] = units[percents].round(3)
        units.to_parquet(self.dataset_paths["units"], compression="gzip")


# Types of variables:
# - bounded above, unknown upper bound (concentration)
# - bounded above, known upper bound (value)
# - bounded above, known upper bound (percent)
# - lower bound =0 ⟶ log transform not applicable
# - lower bound ≠0 ⟶ log transform applicable

_BOUNDS = {
    "Acetate": (0, 2.5),  # concentration like
    "Glucose": (0, 20),  # concentration like
    "DOT": (0, 100),  # percent like
    "OD600": (0, 100),  # percent like
    "FlowAir": (0, None),  # possibly log-transform
    "Base": (0, None),  # possibly log-transform
    "Volume": (0, None),  # possibly log-transform
    "CumulatedFeedGlucose": (0, None),  # possibly log-transform
    "CumulatedFeedMedium": (0, None),  # possibly log-transform
    "ProbeVolume": (0, None),  # possibly log-transform
    "StirringSpeed": (0, None),  # possibly log-transform
    "InducerConcentration": (0, None),  # possibly log-transform
    "FluoGFP": (0, 1_000_000),
    "pH": (4, 10),  # log scale
    "Temperature": (20, 45),
}
