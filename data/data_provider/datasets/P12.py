# Code from: https://github.com/Ladbaby/PyOmniTS
import warnings

import torch
from sklearn.model_selection import train_test_split

from utils.ExpConfigs import ExpConfigs
from data.dependencies.tsdm.tasks.P12 import Physionet2012
from data.dependencies.tsdm.PyOmniTS.tsdmDataset import (
    tsdmDataset, 
    collate_fn, 
    collate_fn_patch, 
    collate_fn_tpatch
) # collate_fns must be imported here for PyOmniTS's --collate_fn argument to work

warnings.filterwarnings('ignore')

class Data(tsdmDataset):
    '''
    wrapper for PhysioNet 2012 dataset implemented in tsdm
    tsdm: https://openreview.net/forum?id=a-bD9-0ycs0

    - title: "Predicting In-Hospital Mortality of ICU Patients: The PhysioNet/Computing in Cardiology Challenge 2012"
    - paper link: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3965265/
    - tasks: forecasting
    - sampling rate (rounded): 1 hour
    - max time length (padded): 48 (48 hours)
    - seq_len -> pred_len:
        - 36 -> 3
        - 36 -> 12
        - 24 -> 24
    - number of variables: 36
        
        - 0: Albumin (g/dL)
        - 1: ALP [Alkaline phosphatase (IU/L)]
        - 2: ALT [Alanine transaminase (IU/L)]
        - 3: AST [Aspartate transaminase (IU/L)]
        - 4: Bilirubin (mg/dL)
        - 5: BUN [Blood urea nitrogen (mg/dL)]
        - 6: Cholesterol (mg/dL)
        - 7: Creatinine [Serum creatinine (mg/dL)]
        - 8: DiasABP [Invasive diastolic arterial blood pressure (mmHg)]
        - 9: FiO2 [Fractional inspired O2 (0-1)]
        - 10: GCS [Glasgow Coma Score (3-15)]
        - 11: Glucose [Serum glucose (mg/dL)]
        - 12: HCO3 [Serum bicarbonate (mmol/L)]
        - 13: HCT [Hematocrit (%)]
        - 14: HR [Heart rate (bpm)]
        - 15: K [Serum potassium (mEq/L)]
        - 16: Lactate (mmol/L)
        - 17: Mg [Serum magnesium (mmol/L)]
        - 18: MAP [Invasive mean arterial blood pressure (mmHg)]
        - 19: MechVent [Mechanical ventilation respiration (0:false, or 1:true)]
        - 20: Na [Serum sodium (mEq/L)]
        - 21: NIDiasABP [Non-invasive diastolic arterial blood pressure (mmHg)]
        - 22: NIMAP [Non-invasive mean arterial blood pressure (mmHg)]
        - 23: NISysABP [Non-invasive systolic arterial blood pressure (mmHg)]
        - 24: PaCO2 [partial pressure of arterial CO2 (mmHg)]
        - 25: PaO2 [Partial pressure of arterial O2 (mmHg)]
        - 26: pH [Arterial pH (0-14)]
        - 27: Platelets (cells/nL)
        - 28: RespRate [Respiration rate (bpm)]
        - 29: SaO2 [O2 saturation in hemoglobin (%)]
        - 30: SysABP [Invasive systolic arterial blood pressure (mmHg)]
        - 31: Temp [Temperature (°C)]
        - 32: TropI [Troponin-I (μg/L)]
        - 33: TropT [Troponin-T (μg/L)]
        - 34: Urine [Urine output (mL)]
        - 35: WBC [White blood cell count (cells/nL)]
    - number of samples: 11981 (9704 + 1078 + 1199)
    '''
    def __init__(
        self, 
        configs: ExpConfigs,
        flag: str = 'train', 
        **kwargs
    ):
        super(Data, self).__init__(configs=configs, flag=flag)
        self.L_TOTAL = 48 # overwrite None in parent class

        self._check_lengths()
        self._preprocess()
        self._get_sample_index() # overwrite self.sample_index=None in parent class

    def __getitem__(self, index): # redundant, just for clarity
        return super().__getitem__(index)

    def __len__(self): # redundant, just for clarity
        return super().__len__()

    def _check_lengths(self): # redundant, just for clarity
        return super()._check_lengths()

    def _preprocess_base(self, task): # redundant, just for clarity
        return super()._preprocess_base(task)

    def _preprocess(self):
        if self.configs.task_name in ["short_term_forecast", "long_term_forecast"]:
            backbone_pred_len = self.pred_len
        else:
            raise NotImplementedError

        task = Physionet2012(
            seq_len=self.seq_len - 0.5,
            pred_len=backbone_pred_len
        )
        self._preprocess_base(task) # implemented in parent class

    def _get_sample_index(self):
        N_SAMPLES = 11981
        sample_index_all = torch.arange(N_SAMPLES)
        sample_index_train_val, sample_index_test = train_test_split(sample_index_all, test_size=0.1, shuffle = False)
        sample_index_train, sample_index_val = train_test_split(sample_index_train_val, test_size=0.1, shuffle = False)
        if self.flag == "train":
            self.sample_index = sample_index_train
        elif self.flag == "val":
            self.sample_index = sample_index_val
        elif self.flag == "test":
            self.sample_index = sample_index_test
        elif self.flag == "test_all":
            self.sample_index = sample_index_all
        else:
            raise NotImplementedError(f"Unknown {self.flag=}")