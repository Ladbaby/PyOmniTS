r"""MIMIC-III Clinical Database.

Abstract
--------
MIMIC-III is a large, freely-available database comprising de-identified health-related
data associated with over forty thousand patients who stayed in critical care units of
the Beth Israel Deaconess Medical Center between 2001 and 2012.
The database includes information such as demographics, vital sign measurements made at
the bedside (~1 data point per hour), laboratory test results, procedures, medications,
caregiver notes, imaging reports, and mortality (including post-hospital discharge).

MIMIC supports a diverse range of analytic studies spanning epidemiology, clinical
decision-rule improvement, and electronic tool development. It is notable for three
factors: it is freely available to researchers worldwide; it encompasses a diverse and
very large population of ICU patients; and it contains highly granular data, including
vital signs, laboratory results, and medications.
"""

__all__ = ["MIMIC_III_DeBrouwer2019"]


from pathlib import Path

import pandas as pd

from data.dependencies.tsdm.datasets.base import MultiFrameDataset
from data.dependencies.tsdm.encoders import TripletDecoder


class MIMIC_III_DeBrouwer2019(MultiFrameDataset):
    r"""MIMIC-III Clinical Database.

    MIMIC-III is a large, freely-available database comprising de-identified health-related data
    associated with over forty thousand patients who stayed in critical care units of the Beth
    Israel Deaconess Medical Center between 2001 and 2012. The database includes information such
    as demographics, vital sign measurements made at the bedside (~1 data point per hour),
    laboratory test results, procedures, medications, caregiver notes, imaging reports, and
    mortality (including post-hospital discharge).

    MIMIC supports a diverse range of analytic studies spanning epidemiology, clinical decision-rule
    improvement, and electronic tool development. It is notable for three factors: it is freely
    available to researchers worldwide; it encompasses a diverse and very large population of ICU
    patients; and it contains highly granular data, including vital signs, laboratory results,
    and medications.

    Notes
    -----
    NOTE: ``TIME_STAMP = round(merged_df_short_binned["TIME_STAMP"].dt.total_seconds()*bin_k/(100*36))``
    and ``bin_k = 10``
    i.e. ``TIME_STAMP = round(dt.total_seconds()*10/3600) = round(dt.total_hours()*10)``
    i.e. ``TIME_STAMP ≈ 10*total_hours``
    so e.g. the last patient was roughly 250 hours, 10½ days.
    """

    BASE_URL = r"https://physionet.org/content/mimiciii/get-zip/1.4/"
    INFO_URL = r"https://physionet.org/content/mimiciii/1.4/"
    HOME_URL = r"https://mimic.mit.edu/"
    GITHUB_URL = r"https://github.com/edebrouwer/gru_ode_bayes/"
    RAWDATA_SHA256 = "8e884a916d28fd546b898b54e20055d4ad18d9a7abe262e15137080e9feb4fc2"
    RAWDATA_SHAPE = (3082224, 7)
    DATASET_SHA256 = {
        "timeseries": "2ebb7da820560f420f71c0b6fb068a46449ef89b238e97ba81659220fae8151b",
        "metadata": "4779aa3639f468126ea263645510d5395d85b73caf1c7abb0a486561b761f5b4",
    }
    DATASET_SHAPE = {"timeseries": (552327, 96), "metadata": (96, 3)}

    dataset_files = {"timeseries": "timeseries.parquet", "metadata": "metadata.parquet"}
    rawdata_files = "complete_tensor.csv"
    rawdata_paths: Path
    index = ["timeseries", "metadata"]

    def _clean(self, key):
        if not self.rawdata_paths.exists():
            raise RuntimeError(
                f"Please apply the preprocessing code found at {self.GITHUB_URL}."
                f"\nPut the resulting file 'complete_tensor.csv' in {self.RAWDATA_DIR}."
            )

        ts = pd.read_csv(self.rawdata_paths, index_col=0)

        if ts.shape != self.RAWDATA_SHAPE:
            raise ValueError(
                f"The {ts.shape=} is not correct."
                "Please apply the modified preprocessing using bin_k=2, as outlined in"
                "the appendix. The resulting tensor should have 3082224 rows and 7 columns."
            )

        ts = ts.sort_values(by=["UNIQUE_ID", "TIME_STAMP"])
        ts = ts.astype(
            {
                "UNIQUE_ID": "int16",
                "TIME_STAMP": "int16",
                "LABEL_CODE": "int16",
                "VALUENORM": "float32",
                "MEAN": "float32",
                "STD": "float32",
            }
        )

        means = ts.groupby("LABEL_CODE").mean()["VALUENUM"].rename("MEANS")
        stdvs = ts.groupby("LABEL_CODE").std()["VALUENUM"].rename("STDVS")
        stats = pd.DataFrame([means, stdvs]).T.reset_index()
        stats = stats.astype(
            {
                "LABEL_CODE": "int16",
                "MEANS": "float32",
                "STDVS": "float32",
            }
        )

        ts = ts[["UNIQUE_ID", "TIME_STAMP", "LABEL_CODE", "VALUENORM"]]
        ts = ts.reset_index(drop=True)
        ts = ts.set_index(["UNIQUE_ID", "TIME_STAMP"])
        ts = ts.sort_index()
        encoder = TripletDecoder(value_name="VALUENORM", var_name="LABEL_CODE")
        encoder.fit(ts)
        ts = encoder.encode(ts)
        ts.columns = ts.columns.astype("string")
        stats.to_parquet(self.dataset_paths["metadata"])
        ts.to_parquet(self.dataset_paths["timeseries"])

    def _load(self, key):
        # return NotImplemented
        return pd.read_parquet(self.dataset_paths[key])

    def _download(self, **kwargs):
        if not self.rawdata_paths.exists():
            raise RuntimeError(
                f"Please apply the preprocessing code found at {self.GITHUB_URL}."
                f"\nPut the resulting file 'complete_tensor.csv' in {self.RAWDATA_DIR}."
            )
