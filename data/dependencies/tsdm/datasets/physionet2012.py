r"""Physionet Challenge 2012.

Physionet Challenge 2012 Data Set
=================================

The development of methods for prediction of mortality rates in Intensive Care Unit (ICU) populations has been
motivated primarily by the need to compare the efficacy of medications, care guidelines, surgery, and other
interventions when, as is common, it is necessary to control for differences in severity of illness or trauma, age,
and other factors. For example, comparing overall mortality rates between trauma units in a community hospital,
a teaching hospital, and a military field hospital is likely to reflect the differences in the patient populations more
than any differences in standards of care. Acuity scores such as APACHE and SAPS-II are widely used to account for
these differences in the context of such studies.

By contrast, the focus of the PhysioNet/CinC Challenge 2012 is to develop methods for patient-specific prediction of in
hospital mortality. Participants will use information collected during the first two days of an ICU stay to predict
which patients survive their hospitalizations, and which patients do not.

Data for the Challenge
----------------------

The data used for the challenge consist of records from 12,000 ICU stays. All patients were
adults who were admitted for a wide variety of reasons to cardiac, medical, surgical, and
trauma ICUs. ICU stays of less than 48 hours have been excluded. Patients with DNR
(do not resuscitate) or CMO (comfort measures only) directives were not excluded.

Four thousand records comprise training set A, and the remaining records form test sets B and C.
Outcomes are provided for the training set records, and are withheld for the test set records.

Up to 42 variables were recorded at least once during the first 48 hours after admission
to the ICU. Not all variables are available in all cases, however. Six of these variables
are general descriptors (collected on admission), and the remainder are time series,
for which multiple observations may be available.

Each observation has an associated time-stamp indicating the elapsed time of the observation since
ICU admission in each case, in hours and minutes. Thus, for example, a time stamp of 35:19 means that
the associated observation was made 35 hours and 19 minutes after the patient was admitted to the ICU.

Each record is stored as a comma-separated value (CSV) text file. To simplify downloading, participants may download
a zip file or tarball containing all of training set A or test set B. Test set C will be used for validation only and
will not be made available to participants.


Update (8 May 2012): The extraneous ages that were present in the previous versions of some data files have been
removed, and a new general descriptor (ICUType, see below) has been added in each data file.

Five additional outcome-related descriptors, described below, are known for each record.
These are stored in separate CSV text files for each of sets A, B, and C, but only those for set A are available to
challenge participants.

All valid values for general descriptors, time series variables, and outcome-related descriptors are non-negative.
A value of -1 indicates missing or unknown data (for example, if a patient's height was not recorded).

General descriptors
-------------------

As noted, these six descriptors are collected at the time the patient is admitted to the ICU.
Their associated time-stamps are set to 00:00 (thus they appear at the beginning of each patient's record).

RecordID (a unique integer for each ICU stay)
Age (years)
Gender (0: female, or 1: male)
Height (cm)
ICUType (1: Coronary Care Unit, 2: Cardiac Surgery Recovery Unit, 3: Medical ICU, or 4: Surgical ICU)
Weight (kg)*.
The ICUType was added for use in Phase 2; it specifies the type of ICU to which the patient has been admitted.

Time Series
-----------

These 37 variables may be observed once, more than once, or not at all in some cases:

- Albumin (g/dL)
- ALP [Alkaline phosphatase (IU/L)]
- ALT [Alanine transaminase (IU/L)]
- AST [Aspartate transaminase (IU/L)]
- Bilirubin (mg/dL)
- BUN [Blood urea nitrogen (mg/dL)]
- Cholesterol (mg/dL)
- Creatinine [Serum creatinine (mg/dL)]
- DiasABP [Invasive diastolic arterial blood pressure (mmHg)]
- FiO2 [Fractional inspired O2 (0-1)]
- GCS [Glasgow Coma Score (3-15)]
- Glucose [Serum glucose (mg/dL)]
- HCO3 [Serum bicarbonate (mmol/L)]
- HCT [Hematocrit (%)]
- HR [Heart rate (bpm)]
- K [Serum potassium (mEq/L)]
- Lactate (mmol/L)
- Mg [Serum magnesium (mmol/L)]
- MAP [Invasive mean arterial blood pressure (mmHg)]
- MechVent [Mechanical ventilation respiration (0:false, or 1:true)]
- Na [Serum sodium (mEq/L)]
- NIDiasABP [Non-invasive diastolic arterial blood pressure (mmHg)]
- NIMAP [Non-invasive mean arterial blood pressure (mmHg)]
- NISysABP [Non-invasive systolic arterial blood pressure (mmHg)]
- PaCO2 [partial pressure of arterial CO2 (mmHg)]
- PaO2 [Partial pressure of arterial O2 (mmHg)]
- pH [Arterial pH (0-14)]
- Platelets (cells/nL)
- RespRate [Respiration rate (bpm)]
- SaO2 [O2 saturation in hemoglobin (%)]
- SysABP [Invasive systolic arterial blood pressure (mmHg)]
- Temp [Temperature (°C)]
- TropI [Troponin-I (μg/L)]
- TropT [Troponin-T (μg/L)]
- Urine [Urine output (mL)]
- WBC [White blood cell count (cells/nL)]
- Weight (kg)*

The time series measurements are recorded in chronological order within each record, and the associated time stamps
indicate the elapsed time since admission to the ICU. Measurements may be recorded at regular intervals ranging from
hourly to daily, or at irregular intervals as required. Not all time series are available in all cases.

In a few cases, such as blood pressure, different measurements made using two or more methods or sensors
may be recorded with the same or only slightly different time-stamps. Occasional outliers should be expected as well.

Note that Weight is both a general descriptor (recorded on admission) and a time series variable
(often measured hourly, for estimating fluid balance).

Outcome-related Descriptors
---------------------------

The outcome-related descriptors are kept in a separate CSV text file for each of the three record sets; as noted, only
the file associated with training set A is available to participants. Each line of the outcomes file contains these
descriptors:

- RecordID (defined as above)
- SAPS-I score (Le Gall et al., 1984)
- SOFA score (Ferreira et al., 2001)
- Length of stay (days)
- Survival (days)
- In-hospital death (0: survivor, or 1: died in-hospital)

The Length of stay is the number of days between the patient's admission to the ICU and the end of hospitalization
(including any time spent in the hospital after discharge from the ICU).
If the patient's death was recorded (in or out of hospital), then Survival is the number of days between ICU admission
and death; otherwise, Survival is assigned the value -1. Since patients who spent less than 48 hours in the ICU have
been excluded, Length of stay and Survival never have the values 0 or 1 in the challenge data sets.
Given these definitions and constraints,

- Survival > Length of stay  =>  Survivor
- Survival = -1  =>  Survivor
- 2 <= Survival <= Length of stay  =>  In-hospital death
"""

__all__ = [
    # Classes
    "Physionet2012",
]

import os
import tarfile
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import IO, Any

import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm

from data.dependencies.tsdm.datasets.base import DATASET_OBJECT, MultiFrameDataset
from data.dependencies.tsdm.encoders import TripletDecoder
from data.dependencies.tsdm.utils.types import PathType
import pdb
GENERAL_DESCRIPTORS = {
    "Age": int,
    "Gender": int,
    "Height": float,
    "ICUType": int,
    "Weight": float,
}

variables = {'pH': np.nan, 'PaCO2': np.nan, 'PaO2': np.nan, 'FiO2': np.nan, 'MechVent': np.nan, 'DiasABP': np.nan, 'HR': np.nan, 'MAP': np.nan,
       'SysABP': np.nan, 'Temp': np.nan, 'GCS': np.nan, 'Urine': np.nan, 'Weight': np.nan, 'HCT': np.nan, 'BUN': np.nan,
       'Creatinine': np.nan, 'Glucose': np.nan, 'HCO3': np.nan, 'Mg': np.nan, 'Platelets': np.nan, 'K': np.nan, 'Na': np.nan,
       'WBC': np.nan, 'NIDiasABP': np.nan, 'NIMAP': np.nan, 'NISysABP': np.nan, 'RespRate': np.nan, 'ALP': np.nan, 'ALT': np.nan,
       'AST': np.nan, 'Bilirubin': np.nan, 'SaO2': np.nan, 'Lactate': np.nan, 'Albumin': np.nan, 'TroponinT': np.nan,
       'Cholesterol': np.nan, 'TroponinI': np.nan}
countn = {'pH': 0, 'PaCO2': 0, 'PaO2': 0, 'FiO2': 0, 'MechVent': 0, 'DiasABP': 0, 'HR': 0, 'MAP': 0,
       'SysABP': 0, 'Temp': 0, 'GCS': 0, 'Urine': 0, 'Weight': 0, 'HCT': 0, 'BUN': 0,
       'Creatinine': 0, 'Glucose': 0, 'HCO3': 0, 'Mg': 0, 'Platelets': 0, 'K': 0, 'Na': 0,
       'WBC': 0, 'NIDiasABP': 0, 'NIMAP': 0, 'NISysABP': 0, 'RespRate': 0, 'ALP': 0, 'ALT': 0,
       'AST': 0, 'Bilirubin': 0, 'SaO2': 0, 'Lactate': 0, 'Albumin': 0, 'TroponinT': 0,
       'Cholesterol': 0, 'TroponinI': 0}
time_range = np.arange(0,49)

def _append_record(d: dict[str, list], r: Mapping) -> dict[str, list]:
    for key in d:
        d[key].append(r[key])

    return d


def read_physionet_record(
    f: IO[bytes], unravel_triplets: bool = False
) -> tuple[int, int, dict, DATASET_OBJECT]:
    r"""Read a single record."""
    lines = iter(f)

    # header
    _ = next(lines)

    general_descriptors = {}
    record_id = None
    ICU_type = None

    timestamps: list[int] = []
    # timeseries: dict[str, Any] = {"Parameter": [], "Value": []}
    timeseries: dict[str, Any] = variables.copy()
    counts = countn.copy()
    # pdb.set_trace()
    df = pd.DataFrame(columns=variables)
    prev_ts = -1.
    first = True
    for line in lines:
        time, parameter_bytes, value_bytes = (
            token.strip() for token in line.split(b",")
        )
        hours, minutes = time.split(b":")
        time_minutes = int(hours) * 60 + int(minutes)
        parameter = parameter_bytes.decode(encoding="utf-8")

        if parameter == "RecordID":
            record_id = int(value_bytes.decode())

        elif parameter in GENERAL_DESCRIPTORS and time_minutes == 0:
            v = GENERAL_DESCRIPTORS[parameter](value_bytes.decode())
            general_descriptors[parameter] = v if v > -1 else np.nan

        else:
            value = float(value_bytes.decode())

            if (prev_ts == (time_minutes//60)) or (first==True):
                if first:
                    prev_ts = time_minutes//60
                # pdb.set_trace()
                if parameter not in timeseries:
                    continue
                if np.isnan(timeseries[parameter]):
                    timeseries[parameter] = 0
                timeseries[parameter] += value
                counts[parameter] += 1
                first = False
            else:

                ts = pd.DataFrame(timeseries, index=[np.int32(prev_ts)])
                counts = pd.DataFrame(counts, index=[np.int32(prev_ts)])

                ts = ts.div(counts)
                # 
                df = pd.concat([df, ts])
                prev_ts = time_minutes//60
                timestamps: list[int] = []
                timeseries: dict[str, Any] = variables.copy()
                counts = countn.copy()
            # timeseries["Parameter"].append(parameter)
            # timeseries["Value"].append(value)

    if record_id is None:
        raise ValueError("RecordID not found")
    for key in GENERAL_DESCRIPTORS:
        assert key in general_descriptors, "incomplete metadata"
    # pdb.set_trace()
    df.index.name = 'Time'
    # parameters = np.array(timeseries["Parameter"], dtype=np.unicode_)
    # values = np.array(timeseries["Value"], dtype=np.float32)
    # df = pd.DataFrame(
    #     {"Parameter": parameters, "Value": values},
    #     index=pd.Index(np.array(timestamps, dtype=np.int32), name="Time"),
    # )

    # if unravel_triplets:
    #     decoder = TripletDecoder(sparse=False, var_name="Parameter", value_name="Value")
    #     decoder.fit(df)
    #     ts = decoder.encode(df)
    # else:
    #     ts = df
    # pdb.set_trace()
    return record_id, ICU_type, general_descriptors, df


class Physionet2012(MultiFrameDataset):
    r"""Physionet Challenge 2012.

    Each training data file provides two tables.
    The first table provides general descriptors of patients:

    +----------+-----+--------+--------+---------+--------+
    | RecordID | Age | Gender | Height | ICUType | Weight |
    +==========+=====+========+========+=========+========+
    | 141834   | 52  | 1.0    | 172.7  | 2       | 73.0   |
    +----------+-----+--------+--------+---------+--------+
    | 133786   | 46  | 0.0    | 157.5  | 1       | 52.3   |
    +----------+-----+--------+--------+---------+--------+
    | 141492   | 84  | 0.0    | 152.4  | 3       | 61.2   |
    +----------+-----+--------+--------+---------+--------+
    | 142386   | 87  | 0.0    | 160.0  | 4       | 73.0   |
    +----------+-----+--------+--------+---------+--------+
    | 142258   | 71  | 1.0    | NaN    | 3       | 72.9   |
    +----------+-----+--------+--------+---------+--------+
    | ...      | ... | ...    | ...    | ...     | ...    |
    +----------+-----+--------+--------+---------+--------+
    | 142430   | 39  | 0.0    | 157.5  | 2       | 65.9   |
    +----------+-----+--------+--------+---------+--------+
    | 134614   | 77  | 0.0    | 165.1  | 1       | 66.6   |
    +----------+-----+--------+--------+---------+--------+
    | 139802   | 57  | 1.0    | NaN    | 4       | NaN    |
    +----------+-----+--------+--------+---------+--------+
    | 136653   | 57  | 1.0    | NaN    | 3       | 103.9  |
    +----------+-----+--------+--------+---------+--------+
    | 136047   | 67  | 1.0    | NaN    | 3       | 169.0  |
    +----------+-----+--------+--------+---------+--------+

    where `RecordID` defines unique ID of an admission.

    The second table contains measurements over time, each column of the table provides
    a sequence of measurements over time (e.g., arterial pH), where the
    header of the column describes the measurement. Each row of the table provides a collection of
    measurements at the same time (e.g., heart rate and oxygen level at the same time) for the same admission.

    The table is formatted in the following way:

    +----------+------+-----+------------+---------+-----+-------------+-----------+-----------+
    | RecordID | Time | BUN | Creatinine | DiasABP | ... | Cholesterol | TroponinT | TroponinI |
    +==========+======+=====+============+=========+=====+=============+===========+===========+
    | 141834   | 27   | NaN | NaN        | NaN     | ... | NaN         | NaN       | NaN       |
    +----------+------+-----+------------+---------+-----+-------------+-----------+-----------+
    |          | 107  | NaN | NaN        | NaN     | ... | NaN         | NaN       | NaN       |
    +----------+------+-----+------------+---------+-----+-------------+-----------+-----------+
    |          | 112  | NaN | NaN        | 77.0    | ... | NaN         | NaN       | NaN       |
    +----------+------+-----+------------+---------+-----+-------------+-----------+-----------+
    |          | 127  | NaN | NaN        | 81.0    | ... | NaN         | NaN       | NaN       |
    +----------+------+-----+------------+---------+-----+-------------+-----------+-----------+
    |          | 142  | NaN | NaN        | 74.0    | ... | NaN         | NaN       | NaN       |
    +----------+------+-----+------------+---------+-----+-------------+-----------+-----------+
    | ...      |      | ... | ...        | ...     | ... | ...         | ...       | ...       |
    +----------+------+-----+------------+---------+-----+-------------+-----------+-----------+
    | 136047   | 2618 | NaN | NaN        | NaN     | ... | NaN         | NaN       | NaN       |
    +----------+------+-----+------------+---------+-----+-------------+-----------+-----------+
    |          | 2678 | NaN | NaN        | NaN     | ... | NaN         | NaN       | NaN       |
    +----------+------+-----+------------+---------+-----+-------------+-----------+-----------+
    |          | 2738 | NaN | NaN        | NaN     | ... | NaN         | NaN       | NaN       |
    +----------+------+-----+------------+---------+-----+-------------+-----------+-----------+
    |          | 2798 | NaN | NaN        | NaN     | ... | NaN         | NaN       | NaN       |
    +----------+------+-----+------------+---------+-----+-------------+-----------+-----------+

    Entries of NaN (not a number) indicate that there was no recorded measurement of a variable at the time.
    """

    BASE_URL = r"https://archive.physionet.org/challenge/2012/"
    r"""HTTP address from where the dataset can be downloaded."""
    INFO_URL = r"https://archive.physionet.org/challenge/2012/"
    r"""HTTP address containing additional information about the dataset."""

    RAWDATA_SHA256 = {
        "set-a.tar.gz": "8cb250f179cd0952b4b9ebcf8954b63d70383131670fac1cfee13deaa13ca920",
        "set-b.tar.gz": "b1637a2a423a8e76f8f087896cfc5fdf28f88519e1f4e874fbda69b2a64dac30",
        "set-c.tar.gz": "a4a56b95bcee4d50a3874fe298bf2998f2ed0dd98a676579573dc10419329ee1",
    }

    rawdata_files = {"A": "set-a.tar.gz", "B": "set-b.tar.gz", "C": "set-c.tar.gz"}
    rawdata_paths: dict[str, Path]
    unravel_triplets: bool

    @property
    def dataset_files(self):
        r"""Map splits into filenames."""
        postfix = "triplet" if self.unravel_triplets else "sparse"
        return {
            "A": f"Physionet2012-set-A-{postfix}.tar",
            "B": f"Physionet2012-set-B-{postfix}.tar",
            "C": f"Physionet2012-set-C-{postfix}.tar",
        }

    def __init__(self, *, unravel_triplets: bool = False):
        self.unravel_triplets = unravel_triplets
        super().__init__()

    @property
    def index(self) -> list:
        r"""Return the index of the dataset."""
        return list(self.dataset_files.keys())

    def _clean(self, key):
        record_ids_list = []
        metadata: dict[str, list[float]] = {key: [] for key in GENERAL_DESCRIPTORS}
        time_series = []

        with (
            tarfile.open(str(self.rawdata_paths[key]), "r:gz") as archive,
            tqdm(archive.getmembers()) as progress_bar,
        ):
            for member in progress_bar:
                progress_bar.set_description(f"Loading patient data {member}")

                if not member.isreg():
                    continue
                with archive.extractfile(member) as record_f:  # type: ignore[union-attr]
                    record_id, ICU_type, descriptors, observations = read_physionet_record(
                        record_f, unravel_triplets=self.unravel_triplets
                    )
                    record_ids_list.append(record_id)

                    for k in metadata:
                        metadata[k].append(descriptors[k])

                    time_series.append(observations)

        record_ids = np.array(record_ids_list, dtype=np.int64)
        record_ids_unraveled = np.repeat(
            record_ids, repeats=[len(ts) for ts in time_series]
        )
        timestamps_unraveled = np.concatenate([ts.index for ts in time_series])

        metadata_df = pd.DataFrame(
            metadata, index=pd.Index(record_ids, name="RecordID")
        )

        time_series_df = pd.concat(
            time_series,
            ignore_index=True,
        )
        time_series_df.set_index(
            pd.MultiIndex.from_arrays(
                (record_ids_unraveled, timestamps_unraveled), names=("RecordID", "Time")
            ),
            inplace=True,
        )
        time_series_df.columns.name = None

        self.dataset[key] = metadata_df, time_series_df
        # pdb.set_trace()

        return metadata_df, time_series_df

    @staticmethod
    def serialize(
        frames: tuple[DATASET_OBJECT, DATASET_OBJECT], path: PathType, /, **kwargs: Any
    ) -> None:
        r"""Store the dataset as a tar archive with two feather dataframes."""
        metadata, series = frames

        with tarfile.open(path, mode="w") as archive:
            with tempfile.TemporaryDirectory(prefix="tsdm") as tmp_dir:
                path_metadata = os.path.join(tmp_dir, "metadata.feather")
                metadata.reset_index().to_feather(path_metadata)

                archive.add(name=path_metadata, arcname="/metadata.feather")

                path_series = os.path.join(tmp_dir, "series.feather")
                series.reset_index().to_feather(path_series)

                archive.add(name=path_series, arcname="/series.feather")

        archive.close()

    @staticmethod
    def deserialize(
        path: PathType, /, *, squeeze: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        r"""Read provided tar archive."""
        with tarfile.open(path, mode="r") as archive:
            with archive.extractfile(archive.getmember("metadata.feather")) as meta_f:  # type: ignore[union-attr]
                metadata = pd.read_feather(meta_f)
            metadata.set_index(keys="RecordID", drop=True, inplace=True)

            with archive.extractfile(archive.getmember("series.feather")) as series_f:  # type: ignore[union-attr]
                series = pd.read_feather(series_f)
            series.set_index(["RecordID", "Time"], drop=True, inplace=True)

            if squeeze:
                metadata = metadata.squeeze()
                series = series.squeeze()

            return series, metadata