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
from data.dependencies.tsdm.tasks.ushcn_debrouwer2019 import USHCN_DeBrouwer2019
from utils.ExpConfigs import ExpConfigs

warnings.filterwarnings('ignore')

class Data(tsdmDataset):
    '''
    wrapper for USHCN DeBrouwer2019 dataset implemented in tsdm
    tsdm: https://openreview.net/forum?id=a-bD9-0ycs0

    - title: "Long-Term Daily and Monthly Climate Records from Stations Across the Contiguous United States (U.S. Historical Climatology Network)"
    - dataset link: https://www.osti.gov/biblio/1394920
    - tasks: forecasting
    - max time length: 337 (4 year)
    - seq_len -> pred_len:
        - 150 -> 3
        - 150 -> 50
    - number of variables: 5
    - number of samples: 1114 (902 + 100 + 112)
    '''
    def __init__(
        self, 
        configs: ExpConfigs,
        flag: str = 'train', 
        **kwargs
    ):
        super(Data, self).__init__(configs=configs, flag=flag)
        self.L_TOTAL = 200 # overwrite None in parent class

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

        task = USHCN_DeBrouwer2019(
            seq_len=self.seq_len - 0.5,
            pred_len=backbone_pred_len
        )
        self._preprocess_base(task) # implemented in parent class

    def _get_sample_index(self):
        N_SAMPLES = 1114
        sample_index_all = torch.arange(N_SAMPLES)
        sample_index_train_val, sample_index_test = train_test_split(sample_index_all, test_size=0.1, random_state=1)
        sample_index_train, sample_index_val = train_test_split(sample_index_train_val, test_size=0.1, random_state=1)
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