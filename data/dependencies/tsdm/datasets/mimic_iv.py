r"""MIMIC-IV clinical dataset.

Abstract
--------
Retrospectively collected medical data has the opportunity to improve patient care through knowledge discovery and
algorithm development. Broad reuse of medical data is desirable for the greatest public good, but data sharing must
be done in a manner which protects patient privacy. The Medical Information Mart for Intensive Care (MIMIC)-III
database provided critical care data for over 40,000 patients admitted to intensive care units at the
Beth Israel Deaconess Medical Center (BIDMC). Importantly, MIMIC-III was deidentified, and patient identifiers
were removed according to the Health Insurance Portability and Accountability Act (HIPAA) Safe Harbor provision.
MIMIC-III has been integral in driving large amounts of research in clinical informatics, epidemiology,
and machine learning. Here we present MIMIC-IV, an update to MIMIC-III, which incorporates contemporary data
and improves on numerous aspects of MIMIC-III. MIMIC-IV adopts a modular approach to data organization,
highlighting data provenance and facilitating both individual and combined use of disparate data sources.
MIMIC-IV is intended to carry on the success of MIMIC-III and support a broad set of applications within healthcare.
"""

__all__ = ["MIMIC_IV"]

import os
import subprocess
from getpass import getpass
from pathlib import Path

import pandas as pd

from data.dependencies.tsdm.datasets.base import MultiFrameDataset


class MIMIC_IV(MultiFrameDataset):
    r"""MIMIC-IV Clinical Database.

    Retrospectively collected medical data has the opportunity to improve patient care through knowledge discovery and
    algorithm development. Broad reuse of medical data is desirable for the greatest public good, but data sharing must
    be done in a manner which protects patient privacy. The Medical Information Mart for Intensive Care (MIMIC)-III
    database provided critical care data for over 40,000 patients admitted to intensive care units at the
    Beth Israel Deaconess Medical Center (BIDMC). Importantly, MIMIC-III was deidentified, and patient identifiers
    were removed according to the Health Insurance Portability and Accountability Act (HIPAA) Safe Harbor provision.
    MIMIC-III has been integral in driving large amounts of research in clinical informatics, epidemiology,
    and machine learning. Here we present MIMIC-IV, an update to MIMIC-III, which incorporates contemporary data
    and improves on numerous aspects of MIMIC-III. MIMIC-IV adopts a modular approach to data organization,
    highlighting data provenance and facilitating both individual and combined use of disparate data sources.
    MIMIC-IV is intended to carry on the success of MIMIC-III and support a broad set of applications within healthcare.
    """

    BASE_URL = r"https://www.physionet.org/content/mimiciv/get-zip/1.0/"
    INFO_URL = r"https://www.physionet.org/content/mimiciv/1.0/"
    HOME_URL = r"https://mimic.mit.edu/"
    GITHUB_URL = r"https://github.com/mbilos/neural-flows-experiments"
    VERSION = r"1.0"
    RAWDATA_SHA256 = "dd226e8694ad75149eed2840a813c24d5c82cac2218822bc35ef72e900baad3d"

    rawdata_files = "mimic-iv-1.0.zip"
    rawdata_paths: Path

    # fmt: off
    dataset_files = {
        "admissions"         : "mimic-iv-1.0/core/admissions.csv.gz",
        "patients"           : "mimic-iv-1.0/core/patients.csv.gz",
        "transfers"          : "mimic-iv-1.0/core/transfers.csv.gz",
        "chartevents"        : "mimic-iv-1.0/icu/chartevents.csv.gz",
        "datetimeevents"     : "mimic-iv-1.0/icu/datetimeevents.csv.gz",
        "d_items"            : "mimic-iv-1.0/icu/d_items.csv.gz",
        "icustays"           : "mimic-iv-1.0/icu/icustays.csv.gz",
        "inputevents"        : "mimic-iv-1.0/icu/inputevents.csv.gz",
        "outputevents"       : "mimic-iv-1.0/icu/outputevents.csv.gz",
        "procedureevents"    : "mimic-iv-1.0/icu/procedureevents.csv.gz",
        "d_hcpcs"            : "mimic-iv-1.0/hosp/d_hcpcs.csv.gz",
        "diagnoses_icd"      : "mimic-iv-1.0/hosp/diagnoses_icd.csv.gz",
        "d_icd_diagnoses"    : "mimic-iv-1.0/hosp/d_icd_diagnoses.csv.gz",
        "d_icd_procedures"   : "mimic-iv-1.0/hosp/d_icd_procedures.csv.gz",
        "d_labitems"         : "mimic-iv-1.0/hosp/d_labitems.csv.gz",
        "drgcodes"           : "mimic-iv-1.0/hosp/drgcodes.csv.gz",
        "emar"               : "mimic-iv-1.0/hosp/emar.csv.gz",
        "emar_detail"        : "mimic-iv-1.0/hosp/emar_detail.csv.gz",
        "hcpcsevents"        : "mimic-iv-1.0/hosp/hcpcsevents.csv.gz",
        "labevents"          : "mimic-iv-1.0/hosp/labevents.csv.gz",
        "microbiologyevents" : "mimic-iv-1.0/hosp/microbiologyevents.csv.gz",
        "pharmacy"           : "mimic-iv-1.0/hosp/pharmacy.csv.gz",
        "poe"                : "mimic-iv-1.0/hosp/poe.csv.gz",
        "poe_detail"         : "mimic-iv-1.0/hosp/poe_detail.csv.gz",
        "prescriptions"      : "mimic-iv-1.0/hosp/prescriptions.csv.gz",
        "procedures_icd"     : "mimic-iv-1.0/hosp/procedures_icd.csv.gz",
        "services"           : "mimic-iv-1.0/hosp/services.csv.gz",
    }
    # fmt: on

    index = list(dataset_files.keys())

    def _clean(self, key):
        ...

    def _load(self, key):
        return pd.read_parquet(self.dataset_paths[key])

    def _download(self, **_):
        cut_dirs = self.BASE_URL.count("/") - 3
        user = input("MIMIC-IV username: ")
        password = getpass(prompt="MIMIC-IV password: ", stream=None)

        os.environ["PASSWORD"] = password

        subprocess.run(
            f"wget --user {user} --password $PASSWORD -c -r -np -nH -N "
            + f"--cut-dirs {cut_dirs} -P '{self.RAWDATA_DIR}' {self.BASE_URL} ",
            shell=True,
            check=True,
        )

        file = self.RAWDATA_DIR / "index.html"
        os.rename(file, self.rawdata_files)
