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

__all__ = ["MIMIC_III"]

import os
import subprocess
from getpass import getpass
from pathlib import Path

from data.dependencies.tsdm.datasets.base import MultiFrameDataset


class MIMIC_III(MultiFrameDataset):
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
    NOTE: TIME_STAMP = round(merged_df_short_binned["TIME_STAMP"].dt.total_seconds()*bin_k/(100*36))
    and bin_k = 10
    i.e. TIME_STAMP = round(dt.total_seconds()*10/3600) = round(dt.total_hours()*10)
    i.e. TIME_STAMP ≈ 10*total_hours
    so e.g. the last patient was roughly 250 hours, 10½ days.
    """

    BASE_URL = r"https://physionet.org/content/mimiciii/get-zip/1.4/"
    INFO_URL = r"https://physionet.org/content/mimiciii/1.4/"
    HOME_URL = r"https://mimic.mit.edu/"
    GITHUB_URL = r"https://github.com/edebrouwer/gru_ode_bayes/"
    VERSION = r"1.0"
    RAWDATA_SHA256 = r"f9917f0f77f29d9abeb4149c96724618923a4725310c62fb75529a2c3e483abd"

    rawdata_files = "mimic-iv-1.0.zip"
    rawdata_paths: Path
    # fmt: off
    dataset_files = {
        "ADMISSIONS"         : "mimic-iii-clinical-database-1.4/ADMISSIONS.csv.gz",
        "CALLOUT"            : "mimic-iii-clinical-database-1.4/CALLOUT.csv.gz",
        "CAREGIVERS"         : "mimic-iii-clinical-database-1.4/CAREGIVERS.csv.gz",
        "CHARTEVENTS"        : "mimic-iii-clinical-database-1.4/CHARTEVENTS.csv.gz",
        "CPTEVENTS"          : "mimic-iii-clinical-database-1.4/CPTEVENTS.csv.gz",
        "DATETIMEEVENTS"     : "mimic-iii-clinical-database-1.4/DATETIMEEVENTS.csv.gz",
        "D_CPT"              : "mimic-iii-clinical-database-1.4/D_CPT.csv.gz",
        "DIAGNOSES_ICD"      : "mimic-iii-clinical-database-1.4/DIAGNOSES_ICD.csv.gz",
        "D_ICD_DIAGNOSES"    : "mimic-iii-clinical-database-1.4/D_ICD_DIAGNOSES.csv.gz",
        "D_ICD_PROCEDURES"   : "mimic-iii-clinical-database-1.4/D_ICD_PROCEDURES.csv.gz",
        "D_ITEMS"            : "mimic-iii-clinical-database-1.4/D_ITEMS.csv.gz",
        "D_LABITEMS"         : "mimic-iii-clinical-database-1.4/D_LABITEMS.csv.gz",
        "DRGCODES"           : "mimic-iii-clinical-database-1.4/DRGCODES.csv.gz",
        "ICUSTAYS"           : "mimic-iii-clinical-database-1.4/ICUSTAYS.csv.gz",
        "INPUTEVENTS_CV"     : "mimic-iii-clinical-database-1.4/INPUTEVENTS_CV.csv.gz",
        "INPUTEVENTS_MV"     : "mimic-iii-clinical-database-1.4/INPUTEVENTS_MV.csv.gz",
        "LABEVENTS"          : "mimic-iii-clinical-database-1.4/LABEVENTS.csv.gz",
        "MICROBIOLOGYEVENTS" : "mimic-iii-clinical-database-1.4/MICROBIOLOGYEVENTS.csv.gz",
        "NOTEEVENTS"         : "mimic-iii-clinical-database-1.4/NOTEEVENTS.csv.gz",
        "OUTPUTEVENTS"       : "mimic-iii-clinical-database-1.4/OUTPUTEVENTS.csv.gz",
        "PATIENTS"           : "mimic-iii-clinical-database-1.4/PATIENTS.csv.gz",
        "PRESCRIPTIONS"      : "mimic-iii-clinical-database-1.4/PRESCRIPTIONS.csv.gz",
        "PROCEDUREEVENTS_MV" : "mimic-iii-clinical-database-1.4/PROCEDUREEVENTS_MV.csv.gz",
        "PROCEDURES_ICD"     : "mimic-iii-clinical-database-1.4/PROCEDURES_ICD.csv.gz",
        "SERVICES"           : "mimic-iii-clinical-database-1.4/SERVICES.csv.gz",
        "TRANSFERS"          : "mimic-iii-clinical-database-1.4/TRANSFERS.csv.gz",
    }
    # fmt: on

    index = list(dataset_files.keys())

    def _clean(self, key):
        raise NotImplementedError

    def _download(self, **_):
        cut_dirs = self.BASE_URL.count("/") - 3
        user = input("MIMIC-III username: ")
        password = getpass(prompt="MIMIC-III password: ", stream=None)

        os.environ["PASSWORD"] = password

        subprocess.run(
            f"wget --user {user} --password $PASSWORD -c -r -np -nH -N "
            + f"--cut-dirs {cut_dirs} -P '{self.RAWDATA_DIR}' {self.BASE_URL} ",
            shell=True,
            check=True,
        )

        file = self.RAWDATA_DIR / "index.html"
        os.rename(file, self.rawdata_files)
