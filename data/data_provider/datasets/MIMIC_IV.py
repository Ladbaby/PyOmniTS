# Code from: https://github.com/Ladbaby/PyOmniTS
import warnings

import torch
from sklearn.model_selection import train_test_split

from data.dependencies.tsdm.PyOmniTS.tsdmDataset import (  # collate_fns must be imported here for PyOmniTS's --collate_fn argument to work
    collate_fn,
    collate_fn_fractal,
    collate_fn_patch,
    collate_fn_tpatch,
    tsdmDataset,
)
from data.dependencies.tsdm.tasks.mimic_iv_bilos2021 import MIMIC_IV_Bilos2021
from utils.ExpConfigs import ExpConfigs

warnings.filterwarnings('ignore')

class Data(tsdmDataset):
    '''
    wrapper for MIMIC IV Bilos2021 dataset implemented in tsdm
    tsdm: https://openreview.net/forum?id=a-bD9-0ycs0

    - title: "MIMIC-IV, a freely accessible electronic health record dataset"
    - paper link: https://www.nature.com/articles/s41597-022-01899-x
    - tasks: forecasting
    - sampling rate (rounded): 1 minute
    - max time length (padded): 971 (48 hours)
    - seq_len -> pred_len:
        - 2160 -> 3
        - 2160 -> 720
    - number of variables: 100
        - 0: PO Intake
        - 1: Dextrose 5%
        - 2: Heparin Sodium
        - 3: Bicarbonate
        - 4: Calcium, Total
        - 5: Chloride
        - 6: Creatinine
        - 7: Magnesium
        - 8: Phosphate
        - 9: Potassium
        - 10: Sodium
        - 11: Urea Nitrogen
        - 12: Hematocrit
        - 13: Hemoglobin
        - 14: MCH
        - 15: MCV
        - 16: Platelet Count
        - 17: RDW
        - 18: Red Blood Cells
        - 19: White Blood Cells
        - 20: PT
        - 21: PTT
        - 22: Anion Gap
        - 23: Glucose
        - 24: Void
        - 25: Heparin Sodium (Prophylaxis)
        - 26: Vancomycin
        - 27: Pantoprazole (Protonix)
        - 28: Solution
        - 29: Hydromorphone (Dilaudid)
        - 30: Insulin - Regular
        - 31: Cefazolin
        - 32: OR Crystalloid Intake
        - 33: Hydralazine
        - 34: NaCl 0.9%
        - 35: Base Excess
        - 36: Calculated Total CO2
        - 37: Lactate
        - 38: pCO2
        - 39: pH
        - 40: pO2
        - 41: Specific Gravity
        - 42: Foley
        - 43: OR Urine
        - 44: OR EBL
        - 45: Emesis
        - 46: Potassium Chloride
        - 47: Piperacillin/Tazobactam (Zosyn)
        - 48: Propofol
        - 49: Pre-Admission/Non-ICU Intake
        - 50: Gastric Meds
        - 51: Furosemide (Lasix)
        - 52: Free Water
        - 53: GT Flush
        - 54: LR
        - 55: Norepinephrine
        - 56: Alanine Aminotransferase (ALT)
        - 57: Alkaline Phosphatase
        - 58: Asparate Aminotransferase (AST)
        - 59: Bilirubin, Total
        - 60: Basophils
        - 61: Eosinophils
        - 62: Lymphocytes
        - 63: Monocytes
        - 64: Neutrophils
        - 65: Albumin
        - 66: Pre-Admission
        - 67: Acetaminophen-IV
        - 68: Magnesium Sulfate
        - 69: Insulin - Humalog
        - 70: Fentanyl
        - 71: Insulin - Glargine
        - 72: Magnesium Sulfate (Bolus)
        - 73: Fentanyl (Concentrate)
        - 74: Calcium Gluconate
        - 75: K Phos
        - 76: Cefepime
        - 77: Piggyback
        - 78: Morphine Sulfate
        - 79: Sterile Water
        - 80: OR Cell Saver Intake
        - 81: Packed Red Blood Cells
        - 82: KCL (Bolus)
        - 83: Albumin 5%
        - 84: Dexmedetomidine (Precedex)
        - 85: Phenylephrine
        - 86: Nitroglycerin
        - 87: Oral Gastric
        - 88: Lorazepam (Ativan)
        - 89: Metoprolol
        - 90: D5 1/2NS
        - 91: Straight Cath
        - 92: TF Residual
        - 93: Ceftriaxone
        - 94: Midazolam (Versed)
        - 95: Nasogastric
        - 96: Pantoprazole (Protonix) Continuous
        - 97: Stool
        - 98: TF Residual Output
        - 99: Fecal Bag

        Note: Originally, there are (0~101) variables, but 54 (Famotidine (Pepcid)) and 77 (Metronidazole) in original index are removed in tsdm, and remaining variables are reindexed here.
    - number of samples: 17874 (14477 + 1609 + 1788)
    '''
    def __init__(
        self, 
        configs: ExpConfigs,
        flag: str = 'train', 
        **kwargs
    ):
        super(Data, self).__init__(configs=configs, flag=flag)
        self.L_TOTAL = 2880 # overwrite None in parent class

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
            raise NotImplementedError()

        task = MIMIC_IV_Bilos2021(
            seq_len=self.seq_len - 0.5,
            pred_len=backbone_pred_len
        )
        self._preprocess_base(task) # implemented in parent class

    def _get_sample_index(self):
        N_SAMPLES = 17874
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