# Code from: https://github.com/Ladbaby/PyOmniTS
import warnings

import torch
from sklearn.model_selection import train_test_split

from utils.ExpConfigs import ExpConfigs
from data.dependencies.tsdm.tasks.mimic_iii_debrouwer2019 import MIMIC_III_DeBrouwer2019
from data.dependencies.tsdm.PyOmniTS.tsdmDataset import (
    tsdmDataset, 
    collate_fn, 
    collate_fn_patch, 
    collate_fn_tpatch 
) # collate_fns must be imported here for PyOmniTS's --collate_fn argument to work

warnings.filterwarnings('ignore')

class Data(tsdmDataset):
    '''
    wrapper for MIMIC III DeBrouwer2019 dataset implemented in tsdm
    tsdm: https://openreview.net/forum?id=a-bD9-0ycs0

    - title: "MIMIC-III, a freely accessible critical care database"
    - paper link: https://www.nature.com/articles/sdata201635
    - tasks: forecasting
    - sampling rate (rounded): 30 minutes
    - max time length (padded): 96 (48 hours)
    - seq_len -> pred_len:
        - 72 -> 3
        - 72 -> 24
        - 48 -> 48
    - number of variables: 96
        - 0: Anion Gap
        - 1: Bicarbonate
        - 2: Calcium, Total
        - 3: Chloride
        - 4: Creatinine
        - 5: Glucose
        - 6: Magnesium
        - 7: Phosphate
        - 8: Potassium
        - 9: Sodium
        - 10: Alkaline Phosphatase
        - 11: Asparate Aminotransferase (AST)
        - 12: Bilirubin, Total
        - 13: Urea Nitrogen
        - 14: Basophils
        - 15: Eosinophils
        - 16: Hematocrit
        - 17: Hemoglobin
        - 18: Lymphocytes
        - 19: MCH
        - 20: MCHC
        - 21: MCV
        - 22: Monocytes
        - 23: Neutrophils
        - 24: Platelet Count
        - 25: RDW
        - 26: Red Blood Cells
        - 27: White Blood Cells
        - 28: PTT
        - 29: Base Excess
        - 30: Calculated Total CO2
        - 31: Lactate
        - 32: pCO2
        - 33: pH
        - 34: pO2
        - 35: PT
        - 36: Alanine Aminotransferase (ALT)
        - 37: Specific Gravity
        - 38: Sodium Chloride 0.9%  Flush Drug
        - 39: D5W Drug
        - 40: Magnesium Sulfate Drug
        - 41: Potassium Chloride Drug
        - 42: Potassium Chloride
        - 43: Calcium Gluconate
        - 44: Magnesium Sulfate
        - 45: Furosemide (Lasix)
        - 46: Insulin - Regular
        - 47: PO Intake
        - 48: Insulin - Humalog
        - 49: OR Crystalloid Intake
        - 50: Morphine Sulfate
        - 51: Insulin - Glargine
        - 52: OR Cell Saver Intake
        - 53: Sterile Water
        - 54: Dextrose 5%
        - 55: LR
        - 56: Piggyback
        - 57: Solution
        - 58: KCL (Bolus)
        - 59: Magnesium Sulfate (Bolus)
        - 60: Nitroglycerin
        - 61: Albumin
        - 62: Foley
        - 63: Chest Tube #1
        - 64: Metoprolol Tartrate Drug
        - 65: Bisacodyl Drug
        - 66: Docusate Sodium Drug
        - 67: Aspirin Drug
        - 68: Packed Red Blood Cells
        - 69: Phenylephrine
        - 70: Gastric Meds
        - 71: GT Flush
        - 72: Hydralazine
        - 73: Midazolam (Versed)
        - 74: Metoprolol
        - 75: D5 1/2NS
        - 76: Void
        - 77: TF Residual
        - 78: OR EBL
        - 79: Albumin 5%
        - 80: Lorazepam (Ativan)
        - 81: Jackson Pratt #1
        - 82: Pre-Admission
        - 83: Pantoprazole Drug
        - 84: Humulin-R Insulin Drug
        - 85: Stool Out Stool
        - 86: Ultrafiltrate Ultrafiltrate
        - 87: Chest Tube #2
        - 88: Heparin Sodium
        - 89: K Phos
        - 90: Norepinephrine
        - 91: Urine Out Incontinent
        - 92: Ostomy (output)
        - 93: Fecal Bag
        - 94: Gastric Gastric Tube
        - 95: Condom Cath
    - number of samples: 21250 (17212 + 1913 + 2125)
    '''
    def __init__(
        self, 
        configs: ExpConfigs,
        flag: str = 'train', 
        **kwargs
    ):
        super(Data, self).__init__(configs=configs, flag=flag)
        self.L_TOTAL = 96 # overwrite None in parent class

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

        task = MIMIC_III_DeBrouwer2019(
            seq_len=self.seq_len - 0.5,
            pred_len=backbone_pred_len
        )
        self._preprocess_base(task) # implemented in parent class

    def _get_sample_index(self):
        N_SAMPLES = 21250
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