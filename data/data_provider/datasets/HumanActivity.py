# Code from: https://github.com/Ladbaby/PyOmniTS
import math

import torch
from torch import Tensor
from torch.utils.data import Dataset
from sklearn import model_selection

from utils.globals import logger
from utils.ExpConfigs import ExpConfigs
from data.dependencies.HumanActivity.HumanActivity import HumanActivity, Activity_time_chunk
from data.dependencies.tsdm.PyOmniTS.tsdmDataset import (
    collate_fn, 
    collate_fn_patch, 
    collate_fn_tpatch 
) # collate_fns must be imported here for PyOmniTS's --collate_fn argument to work

class Data(Dataset):
    '''
    wrapper for Human Activity dataset

    - title: "Localization Data for Person Activity"
    - dataset link: https://archive.ics.uci.edu/dataset/196/localization+data+for+person+activity
    - tasks: forecasting
    - sampling rate (rounded): 1 millisecond
    - max time length (padded): 131 (4000 milliseconds)
    - seq_len -> pred_len:
        - 3000 -> 300
        - 3000 -> 1000
    - number of variables: 12
    - number of samples: 1360 (949 + 193 + 218)
    '''
    def __init__(
        self, 
        configs: ExpConfigs,
        flag: str = 'train', 
        **kwargs
    ):
        self.configs = configs
        assert flag in ["train", "val", "test", "test_all"]
        self.flag = flag

        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        self.L_TOTAL = 4000

        self.dataset_root_path = configs.dataset_root_path

        self.N_SAMPLES = None # set in _preprocess()
        self.N_SAMPLES_TRAIN = None # set in _preprocess()
        self.N_SAMPLES_VAL = None # set in _preprocess()
        self.N_SAMPLES_TEST = None # set in _preprocess()

        self._check_lengths()
        self._preprocess()
        self._get_sample_index()

    def __getitem__(self, index):
        sample_dict: dict[str, Tensor] = self.data[index]
        sample_dict["sample_ID"] = self.sample_index[index]
        sample_dict["_configs"] = self.configs
        sample_dict["_L_TOTAL"] = self.seq_len + self.pred_len # For HumanActivity, it is the sample length , not self.L_TOTAL

        # WARNING: this is not the final input to the model, they should be processed by any collate_fn!
        '''
        contains the following keys:
        - x
        - x_mark
        - x_mask
        - y
        - y_mark
        - y_mask
        - sample_ID
        '''
        return sample_dict

    def __len__(self):
        return len(self.data)

    def _check_lengths(self):
        if self.configs.task_name in ["short_term_forecast", "long_term_forecast"]:
            assert self.seq_len + self.pred_len <= self.L_TOTAL, f"{self.seq_len+self.pred_len=} is too large. Expect the value smaller than self.L_TOTAL"
        else:
            raise NotImplementedError

    def _preprocess(self):
        if self.configs.task_name in ["short_term_forecast", "long_term_forecast"]:
            backbone_pred_len = self.pred_len
        else:
            raise NotImplementedError

        human_activity = HumanActivity(
            root=self.configs.dataset_root_path
        )

        seen_data, test_data = model_selection.train_test_split(human_activity, train_size= 0.9, random_state = 42, shuffle = False)
        train_data, val_data = model_selection.train_test_split(seen_data, train_size= 0.9, random_state = 42, shuffle = False)
        # logger.info(f"Dataset n_samples: {len(human_activity)=} {len(train_data)=} {len(val_data)=} {len(test_data)=}")

        train_data = Activity_time_chunk(
            data=train_data, 
            history=self.seq_len, 
            pred_window=backbone_pred_len
        )
        val_data = Activity_time_chunk(
            data=val_data,
            history=self.seq_len, 
            pred_window=backbone_pred_len
        )
        test_data = Activity_time_chunk(
            data=test_data, 
            history=self.seq_len, 
            pred_window=backbone_pred_len
        )

        self.N_SAMPLES = len(train_data + val_data + test_data)
        self.N_SAMPLES_TRAIN = len(train_data)
        self.N_SAMPLES_VAL = len(val_data)
        self.N_SAMPLES_TEST = len(test_data)

        if self.flag != "val":
            # val set will follow the setting of train set
            # determine the max number of observations along time, among all samples
            test_all_data = train_data + val_data + test_data
            self.seq_len_max_irr = 0
            self.pred_len_max_irr = 0
            self.patch_len_max_irr = 0
            seq_residual_len = 0

            SEQ_LEN = self.configs.seq_len
            PRED_LEN = self.configs.pred_len

            PATCH_LEN = self.configs.patch_len

            for sample in test_all_data:
                if sample["x"].shape[0] > self.seq_len_max_irr:
                    self.seq_len_max_irr = sample["x"].shape[0]
                if sample["y"].shape[0] > self.pred_len_max_irr:
                    self.pred_len_max_irr = sample["y"].shape[0]

                if self.configs.collate_fn == "collate_fn_patch":
                    assert SEQ_LEN % PATCH_LEN == 0, f"seq_len {SEQ_LEN} should be divisible by patch_len {PATCH_LEN}"
                    n_patch: int = SEQ_LEN // PATCH_LEN
                    n_patch_y: int = math.ceil(self.configs.pred_len / PATCH_LEN)

                    patch_i_end_previous = 0
                    for i in range(n_patch):
                        observations = sample["x_mark"] < ((i + 1) * PATCH_LEN / (SEQ_LEN + PRED_LEN))
                        patch_i_end = observations.sum()
                        sample_mask = slice(patch_i_end_previous, patch_i_end)
                        x_patch_i = sample["x"][sample_mask]
                        if len(x_patch_i) > self.patch_len_max_irr:
                            self.patch_len_max_irr = len(x_patch_i)

                        patch_i_end_previous = patch_i_end

                    patch_j_end_previous = 0
                    for j in range(n_patch_y):
                        observations = sample["y_mark"] < (((n_patch + j + 1) * PATCH_LEN) / (SEQ_LEN + PRED_LEN))
                        patch_j_end = observations.sum()
                        sample_mask = slice(patch_j_end_previous, patch_j_end)
                        y_patch_j = sample["y"][sample_mask]
                        if len(y_patch_j) > self.patch_len_max_irr:
                            self.patch_len_max_irr = len(y_patch_j)

                        patch_j_end_previous = patch_j_end

            if self.configs.collate_fn == "collate_fn_patch":
                n_patch: int = SEQ_LEN // PATCH_LEN
                n_patch_y: int = math.ceil(self.configs.pred_len / PATCH_LEN)
                self.seq_len_max_irr = max(self.seq_len_max_irr, self.patch_len_max_irr * n_patch)
                self.pred_len_max_irr = max(self.pred_len_max_irr, self.patch_len_max_irr * n_patch_y)

            # create a new field in global configs to pass information to models
            self.configs.seq_len_max_irr = self.seq_len_max_irr
            self.configs.pred_len_max_irr = self.pred_len_max_irr
            if self.configs.collate_fn in ["collate_fn_patch", "collate_fn_tpatch"]:
                self.configs.patch_len_max_irr = self.patch_len_max_irr
                logger.debug(f"{self.configs.patch_len_max_irr=}")
            logger.debug(f"{self.configs.seq_len_max_irr=}")
            logger.debug(f"{self.configs.pred_len_max_irr=}")

        if self.flag == "test_all":
            # merge the 3 datasets
            self.data = train_data + val_data + test_data
        elif self.flag == "train":
            self.data = train_data
        elif self.flag == "val":
            self.data = val_data
        elif self.flag == "test":
            self.data = test_data

    def _get_sample_index(self):
        sample_index_all = torch.arange(self.N_SAMPLES)
        if self.flag == "train":
            self.sample_index = sample_index_all[:self.N_SAMPLES_TRAIN]
        elif self.flag == "val":
            self.sample_index = sample_index_all[self.N_SAMPLES_TRAIN:self.N_SAMPLES_TRAIN+self.N_SAMPLES_VAL]
        elif self.flag == "test":
            self.sample_index = sample_index_all[self.N_SAMPLES_TRAIN+self.N_SAMPLES_VAL:]
        elif self.flag == "test_all":
            self.sample_index = sample_index_all
        else:
            raise NotImplementedError(f"Unknown {self.flag=}")