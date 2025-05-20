r"""Data set contains electricity consumption of 370 points/clients.

ElectricityLoadDiagrams20112014 Data Set
========================================

+--------------------------------+------------------------+---------------------------+--------+-------------------------+------------+
| **Data Set Characteristics:**  | Time-Series            | **Number of Instances:**  | 370    | **Area:**               | Computer   |
+--------------------------------+------------------------+---------------------------+--------+-------------------------+------------+
| **Attribute Characteristics:** | Real                   | **Number of Attributes:** | 140256 | **Date Donated**        | 2015-03-13 |
+--------------------------------+------------------------+---------------------------+--------+-------------------------+------------+
| **Associated Tasks:**          | Regression, Clustering | **Missing Values?**       | N/A    | **Number of Web Hits:** | 93733      |
+--------------------------------+------------------------+---------------------------+--------+-------------------------+------------+

Source
------

Artur Trindade, artur.trindade '@' elergone.pt, Elergone, NORTE-07-0202-FEDER-038564
Data type: TS
Task: regression, clustering
Number of Instances (records in your data set):370
Number of Attributes (fields within each record):140256

Data Set Information
--------------------

Data set has no missing values.
Values are in kW of each 15 min. To convert values in kWh values must be divided by 4.
Each column represent one client. Some clients were created after 2011. In these cases consumption were considered zero.
All time labels report to Portuguese hour. However all days present 96 measures (24*4). Every year in March time change
day (which has only 23 hours) the values between 1:00 am and 2:00 am are zero for all points. Every year in October time
change day (which has 25 hours) the values between 1:00 am and 2:00 am aggregate the consumption of two hours.

Attribute Information
---------------------

Data set were saved as txt using csv format, using semi colon (;).
First column present date and time as a string with the following format 'yyyy-mm-dd hh:mm:ss'
Other columns present float values with consumption in kW
"""  # pylint: disable=line-too-long # noqa: E501

__all__ = [
    # Classes
    "Electricity",
]

from pathlib import Path
from zipfile import ZipFile

from pandas import DataFrame, read_csv

from data.dependencies.tsdm.datasets.base import SingleFrameDataset


class Electricity(SingleFrameDataset):
    r"""Data set containing electricity consumption of 370 points/clients.

    +--------------------------------+------------------------+---------------------------+--------+-------------------------+------------+
    | **Data Set Characteristics:**  | Time-Series            | **Number of Instances:**  | 370    | **Area:**               | Computer   |
    +--------------------------------+------------------------+---------------------------+--------+-------------------------+------------+
    | **Attribute Characteristics:** | Real                   | **Number of Attributes:** | 140256 | **Date Donated**        | 2015-03-13 |
    +--------------------------------+------------------------+---------------------------+--------+-------------------------+------------+
    | **Associated Tasks:**          | Regression, Clustering | **Missing Values?**       | N/A    | **Number of Web Hits:** | 93733      |
    +--------------------------------+------------------------+---------------------------+--------+-------------------------+------------+
    """  # pylint: disable=line-too-long # noqa: E501

    BASE_URL = r"https://archive.ics.uci.edu/ml/machine-learning-databases/00321/"
    r"""HTTP address from where the dataset can be downloaded."""
    INFO_URL = (
        r"https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014"
    )
    r"""HTTP address containing additional information about the dataset."""

    DATASET_SHA256 = "5fc87215f2b5097ee96d15e382fd360b92757c241a0315a59c82951945860cc9"
    RAWDATA_SHA256 = "f6c4d0e0df12ecdb9ea008dd6eef3518adb52c559d04a9bac2e1b81dcfc8d4e1"
    DATASET_SHAPE = (140256, 370)
    RAWDATA_SHAPE = (140256, 370)

    dataset: DataFrame
    r"""Store cached version of dataset."""
    rawdata_files = "LD2011_2014.txt.zip"
    rawdata_paths: Path

    def _clean(self) -> DataFrame:
        r"""Create DataFrame with 1 column per client and `pandas.DatetimeIndex`."""
        with ZipFile(self.rawdata_paths) as files:
            fname = self.rawdata_paths.with_suffix("").name
            with files.open(fname) as file:
                df = read_csv(
                    file,
                    sep=";",
                    decimal=",",
                    parse_dates=[0],
                    index_col=0,
                    dtype="float32",
                )
        return df.rename_axis(index="time", columns="client")
