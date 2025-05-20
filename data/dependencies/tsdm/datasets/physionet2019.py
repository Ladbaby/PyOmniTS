r"""Physionet Challenge 2019.

Physionet Challenge 2019 Data Set
=================================

The Challenge data repository contains one file per subject (e.g. training/p00101.psv for the
training data). The complete training database (42 MB) consists of two parts: training set A
(20,336 subjects) and B (20,000 subjects).

Each training data file provides a table with measurements over time. Each column of the table
provides a sequence of measurements over time (e.g., heart rate over several hours), where the
header of the column describes the measurement. Each row of the table provides a collection of
measurements at the same time (e.g., heart rate and oxygen level at the same time).

The table is formatted in the following way:

+-----+-------+------+-----+-------------+--------+-------------+
| HR  | O2Sat | Temp | ... | HospAdmTime | ICULOS | SepsisLabel |
+=====+=======+======+=====+=============+========+=============+
| NaN | NaN   | NaN  | ... | -50         | 1      | 0           |
+-----+-------+------+-----+-------------+--------+-------------+
| 86  | 98    | NaN  | ... | -50         | 2      | 0           |
+-----+-------+------+-----+-------------+--------+-------------+
| 75  | NaN   | NaN  | ... | -50         | 3      | 1           |
+-----+-------+------+-----+-------------+--------+-------------+
| 99  | 100   | 35.5 | ... | -50         | 4      | 1           |
+-----+-------+------+-----+-------------+--------+-------------+

There are 40 time-dependent variables HR, O2Sat, Temp ..., HospAdmTime, which are described here.
The final column, SepsisLabel, indicates the onset of sepsis according to the Sepsis-3 definition,
where 1 indicates sepsis and 0 indicates no sepsis. Entries of NaN (not a number) indicate that
there was no recorded measurement of a variable at the time interval.

More details
------------

Data used in the competition is sourced from ICU patients in three separate hospital systems.
Data from two hospital systems will be publicly available; however, one data set will be censored
and used for scoring. The data for each patient will be contained within a single pipe-delimited
text file. Each file will have the same header and each row will represent a single hour's worth
of data. Available patient co-variates consist of Demographics, Vital Signs, and Laboratory values,
which are defined in the tables below.

The following time points are defined for each patient:

tsuspicion

    1. Clinical suspicion of infection identified as the earlier timestamp of IV antibiotics and
       blood cultures within a specified duration.
    2. If antibiotics were given first, then the cultures must have been obtained within 24 hours.
       If cultures were obtained first, then antibiotic must have been subsequently ordered within
       72 hours.
    3. Antibiotics must have been administered for at least 72 consecutive hours to be considered.

tSOFA

    The occurrence of end organ damage as identified by a two-point deterioration in SOFA score
    within a 24h period.

tsepsis

    The onset time of sepsis is the earlier of tsuspicion and tSOFA as long as tSOFA occurs no more
    than 24 hours before or 12 hours after tsuspicion; otherwise, the patient is not marked as a
    sepsis patient. Specifically, if $t_{\text{suspicion}}−24 ≤ t_{\text{SOFA}} ≤ t_{\text{suspicion}}+12$,
    then $t_{\text{sepsis}} = \min(t_{\text{suspicion}}, t_{\text{SOFA}})$.

Table 1: Columns in each training data file. Vital signs (columns 1-8)
HR 	Heart rate (beats per minute)

+------------------+------------------------------------------------------------------+
| O2Sat            | Pulse oximetry (%)                                               |
+==================+==================================================================+
| Temp             | Temperature (Deg C)                                              |
+------------------+------------------------------------------------------------------+
| SBP              | Systolic BP (mm Hg)                                              |
+------------------+------------------------------------------------------------------+
| MAP              | Mean arterial pressure (mm Hg)                                   |
+------------------+------------------------------------------------------------------+
| DBP              | Diastolic BP (mm Hg)                                             |
+------------------+------------------------------------------------------------------+
| Resp             | Respiration rate (breaths per minute)                            |
+------------------+------------------------------------------------------------------+
| EtCO2            | End tidal carbon dioxide (mm Hg)                                 |
+------------------+------------------------------------------------------------------+
| Laboratory       | values (columns 9-34)                                            |
+------------------+------------------------------------------------------------------+
| BaseExcess       | Measure of excess bicarbonate (mmol/L)                           |
+------------------+------------------------------------------------------------------+
| HCO3             | Bicarbonate (mmol/L)                                             |
+------------------+------------------------------------------------------------------+
| FiO2             | Fraction of inspired oxygen (%)                                  |
+------------------+------------------------------------------------------------------+
| pH               | N/A                                                              |
+------------------+------------------------------------------------------------------+
| PaCO2            | Partial pressure of carbon dioxide from arterial blood (mm Hg)   |
+------------------+------------------------------------------------------------------+
| SaO2             | Oxygen saturation from arterial blood (%)                        |
+------------------+------------------------------------------------------------------+
| AST              | Aspartate transaminase (IU/L)                                    |
+------------------+------------------------------------------------------------------+
| BUN              | Blood urea nitrogen (mg/dL)                                      |
+------------------+------------------------------------------------------------------+
| Alkalinephos     | Alkaline phosphatase (IU/L)                                      |
+------------------+------------------------------------------------------------------+
| Calcium          | (mg/dL)                                                          |
+------------------+------------------------------------------------------------------+
| Chloride         | (mmol/L)                                                         |
+------------------+------------------------------------------------------------------+
| Creatinine       | (mg/dL)                                                          |
+------------------+------------------------------------------------------------------+
| Bilirubin_direct | Bilirubin direct (mg/dL)                                         |
+------------------+------------------------------------------------------------------+
| Glucose          | Serum glucose (mg/dL)                                            |
+------------------+------------------------------------------------------------------+
| Lactate          | Lactic acid (mg/dL)                                              |
+------------------+------------------------------------------------------------------+
| Magnesium        | (mmol/dL)                                                        |
+------------------+------------------------------------------------------------------+
| Phosphate        | (mg/dL)                                                          |
+------------------+------------------------------------------------------------------+
| Potassium        | (mmol/L)                                                         |
+------------------+------------------------------------------------------------------+
| Bilirubin_total  | Total bilirubin (mg/dL)                                          |
+------------------+------------------------------------------------------------------+
| TroponinI        | Troponin I (ng/mL)                                               |
+------------------+------------------------------------------------------------------+
| Hct              | Hematocrit (%)                                                   |
+------------------+------------------------------------------------------------------+
| Hgb              | Hemoglobin (g/dL)                                                |
+------------------+------------------------------------------------------------------+
| PTT              | partial thromboplastin time (seconds)                            |
+------------------+------------------------------------------------------------------+
| WBC              | Leukocyte count (count*10^3/µL)                                  |
+------------------+------------------------------------------------------------------+
| Fibrinogen       | (mg/dL)                                                          |
+------------------+------------------------------------------------------------------+
| Platelets        | (count*10^3/µL)                                                  |
+------------------+------------------------------------------------------------------+
| Demographics     | (columns 35-40)                                                  |
+------------------+------------------------------------------------------------------+
| Age              | Years (100 for patients 90 or above)                             |
+------------------+------------------------------------------------------------------+
| Gender           | Female (0) or Male (1)                                           |
+------------------+------------------------------------------------------------------+
| Unit1            | Administrative identifier for ICU unit (MICU)                    |
+------------------+------------------------------------------------------------------+
| Unit2            | Administrative identifier for ICU unit (SICU)                    |
+------------------+------------------------------------------------------------------+
| HospAdmTime      | Hours between hospital admit and ICU admit                       |
+------------------+------------------------------------------------------------------+
| ICULOS           | ICU length-of-stay (hours since ICU admit)                       |
+------------------+------------------------------------------------------------------+
| Outcome          | (column 41)                                                      |
+------------------+------------------------------------------------------------------+
| SepsisLabel      | For sepsis patients, SepsisLabel is 1 if t≥tsepsis−6 and         |
|                  | 0 if t<tsepsis−6. For non-sepsis patients, SepsisLabel is 0.     |
+------------------+------------------------------------------------------------------+
"""

__all__ = [
    # Classes
    "Physionet2019",
]

from functools import cached_property
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from pandas import DataFrame
from tqdm.autonotebook import tqdm

from data.dependencies.tsdm.datasets.base import SingleFrameDataset


class Physionet2019(SingleFrameDataset):
    r"""Physionet Challenge 2019.

    Each training data file provides a table with measurements over time. Each column of the table
    provides a sequence of measurements over time (e.g., heart rate over several hours), where the
    header of the column describes the measurement. Each row of the table provides a collection of
    measurements at the same time (e.g., heart rate and oxygen level at the same time).

    The table is formatted in the following way:

    +-----+-------+------+-----+-------------+--------+-------------+
    | HR  | O2Sat | Temp | ... | HospAdmTime | ICULOS | SepsisLabel |
    +=====+=======+======+=====+=============+========+=============+
    | NaN | NaN   | NaN  | ... | -50         | 1      | 0           |
    +-----+-------+------+-----+-------------+--------+-------------+
    | 86  | 98    | NaN  | ... | -50         | 2      | 0           |
    +-----+-------+------+-----+-------------+--------+-------------+
    | 75  | NaN   | NaN  | ... | -50         | 3      | 1           |
    +-----+-------+------+-----+-------------+--------+-------------+
    | 99  | 100   | 35.5 | ... | -50         | 4      | 1           |
    +-----+-------+------+-----+-------------+--------+-------------+

    There are 40 time-dependent variables HR, O2Sat, Temp, …, HospAdmTime which are described here.
    The final column, SepsisLabel, indicates the onset of sepsis according to the Sepsis-3
    definition, where 1 indicates sepsis and 0 indicates no sepsis. Entries of NaN (not a number)
    indicate that there was no recorded measurement of a variable at the time interval.
    """

    BASE_URL = r"https://archive.physionet.org/users/shared/challenge-2019/"
    r"""HTTP address from where the dataset can be downloaded"""
    INFO_URL = r"https://physionet.org/content/challenge-2019/"
    r"""HTTP address containing additional information about the dataset"""

    DATASET_SHA256 = "1b9c868bd4c91084545ca7f159a500aa9128d07a30b6e4d47a15354029e66efe"
    DATASET_SHAPE = (1552210, 41)
    RAWDATA_SHA256 = {
        "training_setA.zip": "c0def317798312e4facc0f33ac0202b3a34f412052d9096e8b122b4d3ecb7935",
        "training_setB.zip": "8a88d69a5f64bc9a87d869f527fcc2741c0712cb9a7cb1f5cdcb725336b4c8cc",
    }

    rawdata_files: dict[str, str] = {"A": "training_setA.zip", "B": "training_setB.zip"}
    rawdata_paths: dict[str, Path]

    @cached_property
    def units(self) -> DataFrame:
        r"""Metadata for each unit."""
        _units = [
            # Vital signs (columns 1-8)
            ("HR", "Heart rate", "beats per minute"),
            ("O2Sat", "Pulse oximetry", "%"),
            ("Temp", "Temperature", "Deg C"),
            ("SBP", "Systolic BP", "mm Hg"),
            ("MAP", "Mean arterial pressure", "mm Hg"),
            ("DBP", "Diastolic BP", "mm Hg"),
            ("Resp", "Respiration rate", "breaths per minute"),
            ("EtCO2", "End tidal carbon dioxide", "mm Hg"),
            # Laboratory values (columns 9-34)
            ("BaseExcess", "Measure of excess bicarbonate", "mmol/L"),
            ("HCO3", "Bicarbonate", "mmol/L"),
            ("FiO2", "Fraction of inspired oxygen", "%"),
            ("pH", "N/A", "N/A"),
            (
                "PaCO2",
                "Partial pressure of carbon dioxide from arterial blood",
                "mm Hg",
            ),
            ("SaO2", "Oxygen saturation from arterial blood", "%"),
            ("AST", "Aspartate transaminase", "IU/L"),
            ("BUN", "Blood urea nitrogen", "mg/dL"),
            ("Alkalinephos", "Alkaline phosphatase", "IU/L"),
            ("Calcium", "N/A", "mg/dL"),
            ("Chloride", "N/A", "mmol/L"),
            ("Creatinine", "N/A", "mg/dL"),
            ("Bilirubin_direct", "Bilirubin direct", "mg/dL"),
            ("Glucose", "Serum glucose", "mg/dL"),
            ("Lactate", "Lactic acid", "mg/dL"),
            ("Magnesium", "N/A", "mmol/dL"),
            ("Phosphate", "N/A", "mg/dL"),
            ("Potassium", "N/A", "mmol/L"),
            ("Bilirubin_total", "Total bilirubin", "mg/dL"),
            ("TroponinI", "Troponin I", "ng/mL"),
            ("Hct", "Hematocrit", "%"),
            ("Hgb", "Hemoglobin", "g/dL"),
            ("PTT", "partial thromboplastin time", "seconds"),
            ("WBC", "Leukocyte count", "count*10^3/µL"),
            ("Fibrinogen", "N/A", "mg/dL"),
            ("Platelets", "N/A", "count*10^3/µL"),
            # Demographics (columns 35-40)
            ("Age", "Years (100 for patients 90 or above)"),
            ("Gender", "Female (0) or Male (1)", "N/A"),
            ("Unit1", "Administrative identifier for ICU unit", "MICU"),
            ("Unit2", "Administrative identifier for ICU unit", "SICU"),
            ("HospAdmTime", "Hours between hospital admit and ICU admit", "N/A"),
            ("ICULOS", "ICU length-of-stay (hours since ICU admit)", "N/A"),
            # Outcome (column 41)
            (
                "SepsisLabel",
                "For sepsis patients, SepsisLabel is 1 if t≥tsepsis−6 and 0 if t<tsepsis−6. "
                "For non-sepsis patients, SepsisLabel is 0.",
                "N/A",
            ),
        ]

        units = DataFrame(
            _units, columns=["variable", "description", "unit"], dtype="string"
        )
        units = units.replace("N/A", pd.NA)
        units = units.set_index("variable")

        dtypes = {key: "Float32" for key in units.index} | {
            "Gender": "boolean",
            "Unit1": "boolean",
            "Unit2": "boolean",
            "ICULOS": "Int32",
            "SepsisLabel": "boolean",
        }

        units["dtype"] = pd.Series(dtypes)
        return units

    def _get_frame(self, path: Path) -> DataFrame:
        with (
            ZipFile(path) as archive,
            tqdm(archive.namelist()) as progress_bar,
        ):
            frames = {}
            progress_bar.set_description(f"Loading patient data {path.stem}")

            for compressed_file in progress_bar:
                path = Path(compressed_file)
                name = path.stem[1:]
                if path.suffix != ".psv":
                    continue
                with archive.open(compressed_file) as file:
                    df = pd.read_csv(file, sep="|", header=0)
                    frames[name] = df

        self.LOGGER.info("Concatenating DataFrames")
        frame = pd.concat(frames, names=["patient", "time"])
        frame = frame.astype(self.units["dtype"])
        frame.columns.name = "variable"
        return frame

    def _clean(self) -> DataFrame:
        frames = {
            key: self._get_frame(path) for key, path in self.rawdata_paths.items()
        }
        frame = pd.concat(frames, names=["set"])
        return frame
