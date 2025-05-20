r"""#TODO add module summary line.

#TODO add module description.
"""

from __future__ import annotations

__all__ = [
    # Classes
    "USHCN",
]

import importlib
import logging
import os
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Literal

import pandas
from pandas import DataFrame

from data.dependencies.tsdm.datasets.base import MultiFrameDataset

__logger__ = logging.getLogger(__name__)


def with_cluster(func: Callable) -> Callable:
    r"""Run function with ray cluster."""

    @wraps(func)
    def _wrapper(*args, **kwargs):
        if importlib.util.find_spec("ray") is not None:
            ray = importlib.import_module("ray")
            # Only use 80% of the available CPUs.
            num_cpus = max(1, ((os.cpu_count() or 0) * 4) // 5)
            __logger__.warning("Starting ray cluster with num_cpus=%s.", num_cpus)
            ray.init(num_cpus=num_cpus)
            try:
                return func(*args, **kwargs)
            finally:
                __logger__.warning("Tearing down ray cluster.")
                ray.shutdown()
        else:
            __logger__.warning("Ray not found, skipping ray cluster.")
            return func(*args, **kwargs)

    return _wrapper


class USHCN(MultiFrameDataset[Literal["us_daily", "states", "stations"]]):
    r"""UNITED STATES HISTORICAL CLIMATOLOGY NETWORK (USHCN) Daily Dataset.

    U.S. Historical Climatology Network (USHCN) data are used to quantify national and
    regional-scale temperature changes in the contiguous United States (CONUS).
    The dataset provides adjustments for systematic, non-climatic changes that bias
    temperature trends of monthly temperature records of long-term COOP stations.
    USHCN is a designated subset of the NOAA Cooperative Observer Program (COOP)
    Network, with sites selected according to their spatial coverage, record length,
    data completeness, and historical stability.

    .. rubric:: Stations Data

    +----------+---------+-----------+
    | Variable | Columns | Type      |
    +==========+=========+===========+
    | COOP ID  | 1-6     | Character |
    +----------+---------+-----------+
    | YEAR     | 7-10    | Integer   |
    +----------+---------+-----------+
    | MONTH    | 11-12   | Integer   |
    +----------+---------+-----------+
    | ELEMENT  | 13-16   | Character |
    +----------+---------+-----------+
    | VALUE1   | 17-21   | Integer   |
    +----------+---------+-----------+
    | MFLAG1   | 22      | Character |
    +----------+---------+-----------+
    | QFLAG1   | 23      | Character |
    +----------+---------+-----------+
    | SFLAG1   | 24      | Character |
    +----------+---------+-----------+
    |     ⋮    |    ⋮    |     ⋮     |
    +----------+---------+-----------+
    | VALUE31  | 257-261 | Integer   |
    +----------+---------+-----------+
    | MFLAG31  | 262     | Character |
    +----------+---------+-----------+
    | QFLAG31  | 263     | Character |
    +----------+---------+-----------+
    | SFLAG31  | 264     | Character |
    +----------+---------+-----------+

    .. rubric: Station Variables

    - COOP ID	is the U.S. Cooperative Observer Network station identification code.
      Note that the first two digits in the Coop Id correspond to the state.
    - YEAR		is the year of the record.
    - MONTH	is the month of the record.
    - ELEMENT	is the element type. There are five possible values

        - PRCP = precipitation (hundredths of inches)
        - SNOW = snowfall (tenths of inches)
        - SNWD = snow depth (inches)
        - TMAX = maximum temperature (degrees F)
        - TMIN = minimum temperature (degrees F)

    - VALUE1	is the value on the first day of the month (missing = -9999).
    - MFLAG1	is the measurement flag for the first day of the month. There are five possible values:

        - Blank = no measurement information applicable
        - B = precipitation total formed from two 12-hour totals
        - D = precipitation total formed from four six-hour totals
        - H = represents highest or lowest hourly temperature
        - L = temperature appears to be lagged with respect to reported hour of observation
        - P = identified as "missing presumed zero" in DSI 3200 and 3206
        - T = trace of precipitation, snowfall, or snow depth

    - QFLAG1	is the quality flag for the first day of the month. There are fourteen possible values:

        - Blank = did not fail any quality assurance check
        - D = failed duplicate check
        - G = failed gap check
        - I = failed internal consistency check
        - K = failed streak/frequent-value check
        - L = failed check on length of multiday period
        - M = failed megaconsistency check
        - N = failed naught check
        - O = failed climatological outlier check
        - R = failed lagged range check
        - S = failed spatial consistency check
        - T = failed temporal consistency check
        - W = temperature too warm for snow
        - X = failed bounds check
        - Z = flagged as a result of an official Datzilla investigation

    - SFLAG1	is the source flag for the first day of the month. There are fifteen possible values:

        - Blank = No source (e.g., data value missing)
        - 0 = U.S. Cooperative Summary of the Day (NCDC DSI-3200)
        - 6 = CDMP Cooperative Summary of the Day (NCDC DSI-3206)
        - 7 = U.S. Cooperative Summary of the Day -- Transmitted via WxCoder3 (NCDC DSI-3207)
        - A = U.S. Automated Surface Observing System (ASOS) real-time data (since January 1, 2006)
        - B = U.S. ASOS data for October 2000-December 2005 (NCDC DSI-3211)
        - F = U.S. Fort Data
        - G = Official Global Climate Observing System (GCOS) or other government-supplied data
        - H = High Plains Regional Climate Center real-time data
        - M = Monthly METAR Extract (additional ASOS data)
        - N = Community Collaborative Rain, Hail, and Snow (CoCoRaHS)
        - R = NCDC Reference Network Database (Climate Reference Network and Historical Climatology Network-Modernized)
        - S = Global Summary of the Day (NCDC DSI-9618)

    .. rubric:: Stations Meta-Data

    +-------------+---------+-----------+
    | Variable    | Columns | Type      |
    +=============+=========+===========+
    | COOP ID     | 1-6     | Character |
    +-------------+---------+-----------+
    | LATITUDE    | 8-15    | Real      |
    +-------------+---------+-----------+
    | LONGITUDE   | 17-25   | Real      |
    +-------------+---------+-----------+
    | ELEVATION   | 27-32   | Real      |
    +-------------+---------+-----------+
    | STATE       | 34-35   | Character |
    +-------------+---------+-----------+
    | NAME        | 37-66   | Character |
    +-------------+---------+-----------+
    | COMPONENT 1 | 68-73   | Character |
    +-------------+---------+-----------+
    | COMPONENT 2 | 75-80   | Character |
    +-------------+---------+-----------+
    | COMPONENT 3 | 82-87   | Character |
    +-------------+---------+-----------+
    | UTC OFFSET  | 89-90   | Integer   |
    +-------------+---------+-----------+

    .. rubric:: Station Meta-Data Variables

    - COOP_ID		is the U.S. Cooperative Observer Network station identification code. Note that
      the first two digits in the Coop ID correspond to the assigned state number (see Table 1 below).
    - LATITUDE	is latitude of the station (in decimal degrees).
    - LONGITUDE	is the longitude of the station (in decimal degrees).
    - ELEVATION	is the elevation of the station (in meters, missing = -999.9).
    - STATE		is the U.S. postal code for the state.
    - NAME		is the name of the station location.
    - COMPONENT_1	is the Coop Id for the first station (in chronologic order) whose records were
      joined with those of the USHCN site to form a longer time series. "------" indicates "not applicable".
    - COMPONENT_2	is the Coop Id for the second station (if applicable) whose records were joined
      with those of the USHCN site to form a longer time series.
    - COMPONENT_3	is the Coop Id for the third station (if applicable) whose records were joined
      with those of the USHCN site to form a longer time series.
    - UTC_OFFSET	is the time difference between Coordinated Universal Time (UTC) and local standard time
      at the station (i.e., the number of hours that must be added to local standard time to match UTC).
    """

    BASE_URL = "https://cdiac.ess-dive.lbl.gov/ftp/ushcn_daily/"
    r"""HTTP address from where the dataset can be downloaded."""
    INFO_URL = "https://cdiac.ess-dive.lbl.gov/epubs/ndp/ushcn/daily_doc.html"
    r"""HTTP address containing additional information about the dataset."""
    KEYS = Literal["us_daily", "states", "stations"]
    r"""The names of the DataFrames associated with this dataset."""
    RAWDATA_SHA256 = {
        "data_format.txt": "0fecc3670ea4c00d28385b664a9320d45169dbaea6d7ea962b41274ae77b07ca",
        "ushcn-stations.txt": "002a25791b8c48dd39aa63e438c33a4f398b57cfa8bac28e0cde911d0c10e024",
        "station_file_format.txt": "4acc15ec28aed24f25b75405f611bd719c5f36d6a05c36392d95f5b08a3b798b",
        "us.txt.gz": "4cc2223f92e4c8e3bcb00bd4b13528c017594a2385847a611b96ec94be3b8192",
    }
    DATASET_SHA256 = {
        "us_daily": "03ca354b90324f100402c487153e491ec1da53a3e1eda57575750645b44dbe12",
        "states": "388175ed2bcd17253a7a2db2a6bd8ce91db903d323eaea8c9401024cd19af03f",
        "stations": "1c45405915fd7a133bf7b551a196cc59f75d2a20387b950b432165fd2935153b",
    }
    DATASET_SHAPE = {
        "us_daily": (204771562, 5),
        "states": (48, 3),
        "stations": (1218, 9),
    }
    index = ["us_daily", "states", "stations"]
    rawdata_files = {
        "metadata": "data_format.txt",
        "states": None,
        "stations": "ushcn-stations.txt",
        "stations_metadata": "station_file_format.txt",
        "us_daily": "us.txt.gz",
    }
    rawdata_paths: dict[str, Path]

    # def _load(self, key: KEYS = "us_daily", **kwargs: Any) -> DataFrame:
    #     r"""Load the dataset from disk."""
    #     return super()._load(key=key, **kwargs)

    def _clean(self, key: KEYS = "us_daily") -> DataFrame:
        r"""Create the DataFrames.

        Parameters
        ----------
        key: Literal["us_daily", "states", "stations"], default "us_daily"
        """
        if key == "us_daily":
            return self._clean_us_daily()
        if key == "states":
            return self._clean_states()
        if key == "stations":
            return self._clean_stations()

        raise KeyError(f"Unknown key: {key}")

    def _clean_states(self) -> DataFrame:
        state_dtypes = {
            "ID": pandas.CategoricalDtype(ordered=True),
            "Abbr.": pandas.CategoricalDtype(ordered=True),
            "State": pandas.StringDtype(),
        }
        state_codes = self._state_codes
        columns = state_codes.pop(0)
        states = pandas.DataFrame(state_codes, columns=columns)
        states = states.astype(state_dtypes)
        return states
        # states.to_feather(self.dataset_paths["states"])
        # self.__logger__.info("Finished cleaning 'states' DataFrame")

    def _clean_stations(self) -> DataFrame:
        stations_file = self.rawdata_paths["stations"]
        if not stations_file.exists():
            self.download()

        stations_colspecs = {
            "COOP_ID": (1, 6),
            "LATITUDE": (8, 15),
            "LONGITUDE": (17, 25),
            "ELEVATION": (27, 32),
            "STATE": (34, 35),
            "NAME": (37, 66),
            "COMPONENT_1": (68, 73),
            "COMPONENT_2": (75, 80),
            "COMPONENT_3": (82, 87),
            "UTC_OFFSET": (89, 90),
        }

        # pandas wants list[tuple[int, int]], 0 indexed, half open intervals.
        stations_cspecs = [(a - 1, b) for a, b in stations_colspecs.values()]

        stations_dtypes = {
            "COOP_ID": "string",
            "LATITUDE": "float32",
            "LONGITUDE": "float32",
            "ELEVATION": "float32",
            "STATE": "string",
            "NAME": "string",
            "COMPONENT_1": "string",
            "COMPONENT_2": "string",
            "COMPONENT_3": "string",
            "UTC_OFFSET": "timedelta64[h]",
        }

        stations_new_dtypes = {
            "COOP_ID": "category",
            "COMPONENT_1": "Int32",
            "COMPONENT_2": "Int32",
            "COMPONENT_3": "Int32",
            "STATE": "category",
        }

        stations_na_values = {
            "ELEVATION": "-999.9",
            "COMPONENT_1": "------",
            "COMPONENT_2": "------",
            "COMPONENT_3": "------",
        }

        stations = pandas.read_fwf(
            stations_file,
            colspecs=stations_cspecs,
            dtype=stations_dtypes,
            names=stations_colspecs,
            na_value=stations_na_values,
        )

        for col, na_value in stations_na_values.items():
            stations[col] = stations[col].replace(na_value, pandas.NA)

        stations = stations.astype(stations_new_dtypes)
        stations = stations.set_index("COOP_ID")

        return stations
        # stations.to_parquet(self.dataset_paths["stations"])

    @with_cluster
    def _clean_us_daily(self) -> DataFrame:
        if importlib.util.find_spec("modin") is not None:
            mpd = importlib.import_module("modin.pandas")
        else:
            mpd = pandas

        # column: (start, stop)
        colspecs: dict[str | tuple[str, int], tuple[int, int]] = {
            "COOP_ID": (1, 6),
            "YEAR": (7, 10),
            "MONTH": (11, 12),
            "ELEMENT": (13, 16),
        }

        for k, i in enumerate(range(17, 258, 8)):
            colspecs |= {
                ("VALUE", k + 1): (i, i + 4),
                ("MFLAG", k + 1): (i + 5, i + 5),
                ("QFLAG", k + 1): (i + 6, i + 6),
                ("SFLAG", k + 1): (i + 7, i + 7),
            }

        MFLAGS = pandas.CategoricalDtype(list("BDHKLOPTW"))
        QFLAGS = pandas.CategoricalDtype(list("DGIKLMNORSTWXZ"))
        SFLAGS = pandas.CategoricalDtype(list("067ABFGHKMNRSTUWXZ"))
        ELEMENTS = pandas.CategoricalDtype(("PRCP", "SNOW", "SNWD", "TMAX", "TMIN"))

        dtypes = {
            "COOP_ID": "string",
            "YEAR": "int16",
            "MONTH": "int8",
            "ELEMENT": ELEMENTS,
            "VALUE": pandas.Int16Dtype(),
            "MFLAG": MFLAGS,
            "QFLAG": QFLAGS,
            "SFLAG": SFLAGS,
        }

        # dtypes but with same index as colspec.
        dtype = {
            key: (dtypes[key[0]] if isinstance(key, tuple) else dtypes[key])
            for key in colspecs
        }

        # pandas wants list[tuple[int, int]], 0 indexed, half open intervals.
        cspec = [(a - 1, b) for a, b in colspecs.values()]

        # per column values to be interpreted as nan
        na_values = {("VALUE", k): "-9999" for k in range(1, 32)}
        us_daily_path = self.rawdata_paths["us_daily"]

        self.LOGGER.info("Loading main file...")
        ds = mpd.read_fwf(
            us_daily_path,
            colspecs=cspec,
            names=colspecs,
            na_values=na_values,
            dtype=dtype,
            compression="gzip",
        )

        self.LOGGER.info("Cleaning up columns...")
        # convert data part (VALUES, SFLAGS, MFLAGS, QFLAGS) to stand-alone dataframe
        id_cols = ["COOP_ID", "YEAR", "MONTH", "ELEMENT"]
        data_cols = [col for col in ds.columns if col not in id_cols]
        columns = mpd.DataFrame(data_cols, columns=["VAR", "DAY"])
        columns = columns.astype({"VAR": "string", "DAY": "uint8"})
        columns = columns.astype("category")
        # Turn tuple[VALUE/FLAG, DAY] indices to multi-index:
        data = ds[data_cols]
        data.columns = pandas.MultiIndex.from_frame(columns)

        self.LOGGER.info("Stacking on FLAGS and VALUES columns...")
        # stack on day, this will collapse (VALUE1, ..., VALUE31) into a single VALUE column.
        data = data.stack(level="DAY", dropna=False).reset_index(level="DAY")

        self.LOGGER.info("Merging on ID columns...")
        # correct dtypes after stacking operation
        _dtypes = {k: v for k, v in dtypes.items() if k in data.columns}
        data = data.astype(_dtypes | {"DAY": "int8"})

        # recombine data columns with original data
        data = ds[id_cols].join(data, how="inner")
        data = data.astype(dtypes | {"DAY": "int8", "COOP_ID": "category"})

        self.LOGGER.info("Creating time index...")
        data = data.reset_index(drop=True)
        datetimes = mpd.to_datetime(data[["YEAR", "MONTH", "DAY"]], errors="coerce")
        data = data.drop(columns=["YEAR", "MONTH", "DAY"])
        data["TIME"] = datetimes
        data = data.dropna(subset=["TIME"])

        self.LOGGER.info("Pre-Sorting index....")
        data = data.set_index("COOP_ID")
        data = data.sort_index()  # fast pre-sort with single index
        data = data.set_index("TIME", append=True)

        self.LOGGER.info("Converting back to standard pandas DataFrame....")
        try:
            data = data._to_pandas()  # pylint: disable=protected-access
        except AttributeError:
            pass

        self.LOGGER.info("Sorting columns....")
        data = data.reindex(
            columns=[
                "ELEMENT",
                "MFLAG",
                "QFLAG",
                "SFLAG",
                "VALUE",
            ]
        )
        self.LOGGER.info("Sorting index....")
        data = data.sort_values(by=["COOP_ID", "TIME", "ELEMENT"])

        return data

    @property
    def _state_codes(self):
        return [
            ("ID", "Abbr.", "State"),
            ("01", "AL", "Alabama"),
            ("02", "AZ", "Arizona"),
            ("03", "AR", "Arkansas"),
            ("04", "CA", "California"),
            ("05", "CO", "Colorado"),
            ("06", "CT", "Connecticut"),
            ("07", "DE", "Delaware"),
            ("08", "FL", "Florida"),
            ("09", "GA", "Georgia"),
            ("10", "ID", "Idaho"),
            ("11", "IL", "Idaho"),
            ("12", "IN", "Indiana"),
            ("13", "IA", "Iowa"),
            ("14", "KS", "Kansas"),
            ("15", "KY", "Kentucky"),
            ("16", "LA", "Louisiana"),
            ("17", "ME", "Maine"),
            ("18", "MD", "Maryland"),
            ("19", "MA", "Massachusetts"),
            ("20", "MI", "Michigan"),
            ("21", "MN", "Minnesota"),
            ("22", "MS", "Mississippi"),
            ("23", "MO", "Missouri"),
            ("24", "MT", "Montana"),
            ("25", "NE", "Nebraska"),
            ("26", "NV", "Nevada"),
            ("27", "NH", "NewHampshire"),
            ("28", "NJ", "NewJersey"),
            ("29", "NM", "NewMexico"),
            ("30", "NY", "NewYork"),
            ("31", "NC", "NorthCarolina"),
            ("32", "ND", "NorthDakota"),
            ("33", "OH", "Ohio"),
            ("34", "OK", "Oklahoma"),
            ("35", "OR", "Oregon"),
            ("36", "PA", "Pennsylvania"),
            ("37", "RI", "RhodeIsland"),
            ("38", "SC", "SouthCarolina"),
            ("39", "SD", "SouthDakota"),
            ("40", "TN", "Tennessee"),
            ("41", "TX", "Texas"),
            ("42", "UT", "Utah"),
            ("43", "VT", "Vermont"),
            ("44", "VA", "Virginia"),
            ("45", "WA", "Washington"),
            ("46", "WV", "WestVirginia"),
            ("47", "WI", "Wisconsin"),
            ("48", "WY", "Wyoming"),
        ]
