r"""15 months worth of daily data (440 daily records) that describes the occupancy rate, between 0 and 1, of different car lanes of the San Francisco bay area freeways across time.

+---------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
| **Data Set Characteristics:**   | Multivariate, Time-Series | **Number of Instances:**  | 440    | **Area:**               | Computer   |
+---------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
| **Attribute Characteristics:**  | Real                      | **Number of Attributes:** | 138672 | **Date Donated**        | 2011-05-22 |
+---------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
| **Associated Tasks:**           | Classification            | **Missing Values?**       | N/A    | **Number of Web Hits:** | 79749      |
+---------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+

Source
------

Source: California Department of Transportation, www.pems.dot.ca.gov
Creator: Marco Cuturi, Kyoto University, mcuturi '@' i.kyoto-u.ac.jp

Data Set Information
--------------------

We have downloaded 15 months worth of daily data from the California Department of Transportation PEMS website, [Web Link], The data describes the occupancy
rate, between 0 and 1, of different car lanes of San Francisco bay area freeways. The measurements cover the period from Jan. 1st 2008 to Mar. 30th 2009 and are sampled every 10 minutes. We consider each day in this database as a single time series of dimension 963 (the number of sensors which functioned consistently throughout the studied period) and length 6 x 24=144. We remove public holidays from the dataset, as well
as two days with anomalies (March 8th 2009 and March 9th 2008) where all sensors were muted between 2:00 and 3:00 AM. This results in a database of 440 time series.

The task we propose on this dataset is to classify each observed day as the correct day of the week, from Monday to Sunday, e.g. label it with an integer in {1,2,3,4,5,6,7}.

I will keep separate copies of this database on my website in a Matlab format. If you use Matlab, it might be more convenient to consider these .mat files directly.

Data-Format
-----------

There are two files for each fold, the data file and the labels file. We have key the 440 time series between train and test folds, but you are of course free to merge them to consider a different cross validation setting.
- The PEMS_train textfile has 263 lines. Each line describes a time-series provided as a matrix. The matrix syntax is that of Matlab, e.g. [ a b ; c d] is the matrix with row vectors [a b] and [c d] in that order. Each matrix describes the different occupancies rates (963 lines, one for each station/detector) sampled every 10 minutes during the day (144 columns).
- The PEMS_trainlabel text describes, for each day of measurements described above, the day of the week on which the data was sampled, namely an integer between 1 (Mon.) and 7 (Sun.).

- PEMS_test and PEMS_testlabels are formatted in the same way, except that there are 173 test instances.

- The permutation that I used to shuffle the dataset is given in the randperm file. If you need to rearrange the data so that it follows the calendar order, you should merge train and test samples and reorder them using the inverse permutation of randperm.

Attribute Information
---------------------

Each attribute describes the measurement of the occupancy rate (between 0 and 1) of a captor location as recorded by a measuring station, at a given timestamp in time during the day. The ID of each station is given in the stations_list text file. For more information on the location (GPS, Highway, Direction) of each station please refer to the PEMS website. There are 963 (stations) x 144 (timestamps) = 138.672 attributes for each record.

Relevant Papers
---------------

M. Cuturi, Fast Global Alignment Kernels, Proceedings of the Intern. Conference on Machine Learning 2011.
"""  # pylint: disable=line-too-long # noqa: E501

__all__ = ["Traffic"]

from io import StringIO
from pathlib import Path
from typing import Literal
from zipfile import ZipFile

import numpy as np
import pandas
from pandas import DataFrame, Series

from data.dependencies.tsdm.datasets.base import MultiFrameDataset


def _reformat(s: str, replacements: dict) -> str:
    r"""Replace multiple substrings via dict.

    https://stackoverflow.com/a/64500851/9318372
    """
    *_, result = (s := s.replace(c, r) for c, r in replacements.items())  # noqa: F841
    return result


class Traffic(MultiFrameDataset):
    r"""15 months worth of daily data (440 daily records) that describes the occupancy rate, between 0 and 1, of different car lanes of the San Francisco bay area freeways across time.

    +---------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
    | **Data Set Characteristics:**   | Multivariate, Time-Series | **Number of Instances:**  | 440    | **Area:**               | Computer   |
    +---------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
    | **Attribute Characteristics:**  | Real                      | **Number of Attributes:** | 138672 | **Date Donated**        | 2011-05-22 |
    +---------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
    | **Associated Tasks:**           | Classification            | **Missing Values?**       | N/A    | **Number of Web Hits:** | 79749      |
    +---------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
    """  # pylint: disable=line-too-long # noqa: E501

    BASE_URL = r"https://archive.ics.uci.edu/ml/machine-learning-databases/00204/"
    r"""HTTP address from where the dataset can be downloaded."""
    INFO_URL = r"https://archive.ics.uci.edu/ml/datasets/PEMS-SF"
    r"""HTTP address containing additional information about the dataset."""
    KEYS = Literal["timeseries", "labels", "randperm", "invperm"]
    r"""The names of the DataFrames associated with this dataset."""
    RAWDATA_SHA256 = "371d15048b5401026396d4587e5f9be79792e06d74f7a42a0ec84975e692147e"
    DATASET_SHA256 = {
        "timeseries": "acb7f2a37e14691d67a325e18eecf88c22bc4c175f1a11b5566a07fdf2cd8f62",
        "labels": "c26dc7683548344c5b71ef30d551b6e3f0e726e0d505f45162fde167de7b51cf",
        "randperm": "4d8fa113fd20e397b2802bcc851a8dca861d3e8b806be490a6dff3e0c112f613",
        "invperm": "2838f7df33a292830acf09a3870b495ca0e5524f085aea0b66452248012c9817",
    }
    index: list[KEYS] = ["timeseries", "labels", "randperm", "invperm"]
    r"""The identifiers for the dataset."""
    rawdata_files = "PEMS-SF.zip"
    r"""The name of the zip file containing the raw data."""
    rawdata_paths: Path
    timeseries: DataFrame
    labels: DataFrame
    randperm: DataFrame
    invperm: DataFrame

    def _clean(self, key: KEYS) -> None:
        r"""Create the DataFrames.

        Parameters
        ----------
        key: Literal["us_daily", "states", "stations"], default "us_daily"
        """
        if key == "timeseries":
            return self._clean_data()
        if key == "labels":
            return self._clean_data()
        if key == "randperm":
            return self._clean_randperm()
        if key == "invperm":
            return self._clean_randperm()

        raise KeyError(f"{key} is not a valid key")

    def _clean_data(self, *, use_corrected_dates: bool = True) -> None:
        r"""Create DataFrame from raw data.

        Parameters
        ----------
        use_corrected_dates: bool (default True)
            Use correct dates and anomalies found through reverse engineering the dataset.

        Notes
        -----
        Sampling rate = 10 minutes => 144 samples/day

        PEMS_train: 267 rows
           - each row is data for 1 day.
           - each row encodes a 963×144 matrix (stations×timestamps)

        PEMS_test: same but only 172 rows
        station_labels: labels of the 963 stations

        In total 440 days of observations.
          - original data range is 455 days: 2008-01-01 - 2009-03-30 (15 months)
          - authors manually removed holidays as well as 2 anomalies: 2009-03-08 and 2008-03-09.
          - in total 10 days missing.

        The authors of N-BEATS guesstimate the missing days to be:
            1. Jan. 1, 2008 (New Year’s Day)
            2. Jan. 21, 2008 (Martin Luther King Jr. Day)
            3. Feb. 18, 2008 (Washington’s Birthday)
            4. Mar. 9, 2008 (Anomaly day)
            5. May 26, 2008 (Memorial Day)
            6. Jul. 4, 2008 (Independence Day)
            7. Sep. 1, 2008 (Labor Day)
            8. Oct. 13, 2008 (Columbus Day)
            9. Nov. 11, 2008 (Veterans Day)
            10. Nov. 27, 2008 (Thanksgiving)
            11. Dec. 25, 2008 (Christmas Day)
            12. Jan. 1, 2009 (New Year’s Day)
            13. Jan. 19, 2009 (Martin Luther King Jr. Day)
            14. Feb. 16, 2009 (Washington’s Birthday)
            15. Mar. 8, 2009 (Anomaly day)

        The true missing dates appear to be, by reverse-engineering:
            - "2008-01-02": "1 day off New Year’s Day",
            - "2008-01-22": "1 day off Martin Luther King Jr. Day",
            - "2008-02-19": "1 day off Washington’s Birthday",
            - "2008-03-10": "1 day off anomaly + wrong year",
            - "2008-05-27": "1 day off Memorial Day",
            - "2008-07-05": "1 day off Independence Day",
            - "2008-09-02": "1 day off Labor Day",
            - "2008-10-21": "???",
            - "2008-11-18": "???",
            - "2008-12-08": "???",
            - "2009-02-24": "???",
        """
        # The true anomalies were found by iteratively adding them 1 by one,
        # Each time checking when the first date was when
        # labels[invperm].map(weekdays) didn't match with dates.day_name()
        true_dates = pandas.date_range("2008-01-01", "2009-03-26", freq="d", name="day")
        true_anomalies = pandas.DatetimeIndex(
            {
                "2008-01-01": "New Year’s Day",
                "2008-01-21": "Martin Luther King Jr. Day",
                "2008-02-18": "Washington’s Birthday",
                "2008-03-09": "anomaly + wrong year",
                "2008-05-26": "Memorial Day",
                "2008-07-04": "Independence Day",
                "2008-09-01": "Labor Day",
                "2008-10-20": "???",
                "2008-11-17": "???",
                "2008-12-07": "???",
                "2009-02-23": "???",
            }
        )
        true_weekdays = {
            "1": "Sunday",
            "2": "Monday",
            "3": "Tuesday",
            "4": "Wednesday",
            "5": "Thursday",
            "6": "Friday",
            "7": "Saturday",
        }

        false_dates = pandas.date_range(
            "2008-01-01", "2009-03-30", freq="d", name="day"
        )
        false_anomalies = pandas.DatetimeIndex(
            {
                "Jan. 1, 2008": "New Year’s Day",
                "Jan. 21, 2008": "Martin Luther King Jr. Day",
                "Feb. 18, 2008": "Washington’s Birthday",
                "Mar. 9, 2008": "Anomaly day",
                "May 26, 2008": "Memorial Day",
                "Jul. 4, 2008": "Independence Day",
                "Sep. 1, 2008": "Labor Day",
                "Oct. 13, 2008": "Columbus Day",
                "Nov. 11, 2008": "Veterans Day",
                "Nov. 27, 2008": "Thanksgiving",
                "Dec. 25, 2008": "Christmas Day",
                "Jan. 1, 2009": "New Year’s Day",
                "Jan. 19, 2009": "Martin Luther King Jr. Day",
                "Feb. 16, 2009": "Washington’s Birthday",
                "Mar. 8, 2009": "Anomaly day",
            }
        )
        false_weekdays = {
            "1": "Monday",
            "2": "Tuesday",
            "3": "Wednesday",
            "4": "Thursday",
            "5": "Friday",
            "6": "Saturday",
            "7": "Sunday",
        }

        dates = true_dates if use_corrected_dates else false_dates
        anomalies = true_anomalies if use_corrected_dates else false_anomalies
        weekdays = true_weekdays if use_corrected_dates else false_weekdays

        # remove anomalies
        mask = dates.isin(anomalies)
        assert sum(mask) == len(anomalies)
        dates = dates[~mask]

        # Shuffle dates according to permutation the authors applied
        shuffled_dates = dates[self.randperm]

        timestamps = pandas.timedelta_range(
            "0:00:00", "23:59:59", freq="10min", name="time"
        )
        assert len(timestamps) == 144

        with ZipFile(self.rawdata_paths) as files:
            with files.open("stations_list") as file:
                content = file.read().decode("utf8")
                content = _reformat(content, {"[": "", "]": "", " ": "\n"})
                stations = pandas.read_csv(
                    StringIO(content),
                    names=["station"],
                    dtype="category",
                ).squeeze()
                stations = Series(stations)  # make sure it's not TextFileReader

            with files.open("PEMS_trainlabels") as file:
                content = file.read().decode("utf8")
                content = _reformat(content, {"[": "", "]": "\n", " ": "\n"})
                trainlabels = pandas.read_csv(
                    StringIO(content),
                    names=["labels"],
                    dtype="category",
                ).squeeze()
                train_dates = shuffled_dates[: len(trainlabels)]
                trainlabels.index = train_dates
                trainlabels = Series(trainlabels)  # make sure it's not TextFileReader
            # Check that the labels match with the actual weekdays
            assert (
                trainlabels.index.day_name() == trainlabels.values.map(weekdays)
            ).all(), "Labels do not match with dates!"

            with files.open("PEMS_testlabels") as file:
                content = file.read().decode("utf8")
                content = _reformat(content, {"[": "", "]": "", " ": "\n"})
                testlabels = pandas.read_csv(
                    StringIO(content),
                    names=["labels"],
                    dtype="category",
                ).squeeze()
                test_dates = shuffled_dates[len(trainlabels) :]
                testlabels.index = test_dates
                testlabels = Series(testlabels)  # make sure it's not TextFileReader

            # Check that the labels match with the actual weekdays
            assert (
                testlabels.index.day_name() == testlabels.values.map(weekdays)
            ).all(), "Labels do not match with dates!"
            assert (
                trainlabels.dtype == testlabels.dtype
            ), "Train and test have different labels!"

            with files.open("PEMS_train") as file:
                _PEMS_train = []
                for line in file:
                    content = line.decode("utf8")
                    content = _reformat(
                        content, {"[": "", "]": "", ";": "\n", " ": ","}
                    )
                    df = pandas.read_csv(
                        StringIO(content),
                        header=None,
                    ).squeeze()
                    df = DataFrame(df.values, index=stations, columns=timestamps)
                    _PEMS_train.append(df.T)
                PEMS_train = pandas.concat(_PEMS_train, keys=train_dates)

            with files.open("PEMS_test") as file:
                _PEMS_test = []
                for line in file:
                    content = line.decode("utf8")
                    content = _reformat(
                        content, {"[": "", "]": "", ";": "\n", " ": ","}
                    )
                    df = pandas.read_csv(
                        StringIO(content),
                        header=None,
                    ).squeeze()
                    df = DataFrame(df.values, index=stations, columns=timestamps)
                    _PEMS_test.append(df.T)
                PEMS_test = pandas.concat(_PEMS_test, keys=test_dates)

        PEMS = pandas.concat([PEMS_train, PEMS_test])
        labels = pandas.concat([trainlabels, testlabels]).rename("labels")

        mismatches = labels[self.invperm].map(weekdays) != dates.day_name()
        assert len(dates[mismatches]) == 0, "Mismatches in label and date weekday!"

        PEMS = PEMS.reset_index()

        PEMS["time"] = PEMS["day"] + PEMS["time"]
        PEMS = PEMS.drop(columns="day")
        PEMS = PEMS.set_index("time")
        PEMS = PEMS.astype("float32")
        PEMS.columns = PEMS.columns.astype("string")
        PEMS.to_parquet(self.dataset_paths["timeseries"], compression="gzip")
        DataFrame(labels).to_parquet(self.dataset_paths["labels"], compression="gzip")

    def _clean_randperm(self):
        with ZipFile(self.rawdata_paths) as files:
            with files.open("randperm") as file:
                content = file.read().decode("utf8")
                content = _reformat(content, {"[": "", "]": "", " ": "\n"})
                randperm = pandas.read_csv(
                    StringIO(content),
                    names=["randperm"],
                    dtype="uint16",
                ).squeeze()
                randperm -= 1  # we use 0-based indexing
                invperm = randperm.copy().argsort()
                invperm.name = "invperm"
                assert (randperm[invperm] == np.arange(len(randperm))).all()

        DataFrame(randperm).to_parquet(
            self.dataset_paths["randperm"], compression="gzip"
        )
        DataFrame(invperm).to_parquet(self.dataset_paths["invperm"], compression="gzip")
