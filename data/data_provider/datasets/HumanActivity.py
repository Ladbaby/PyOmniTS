# Code from: https://github.com/Ladbaby/PyOmniTS
import math

import torch
from sklearn import model_selection
from torch import Tensor
from torch.utils.data import Dataset

from data.dependencies.HumanActivity.HumanActivity import (
    Activity_time_chunk,
    HumanActivity,
)
from data.dependencies.tsdm.PyOmniTS.tsdmDataset import (  # collate_fns must be imported here for PyOmniTS's --collate_fn argument to work
    collate_fn,
    collate_fn_fractal,
    collate_fn_patch,
    collate_fn_tpatch,
)
from utils.ExpConfigs import ExpConfigs
from utils.globals import logger


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
        if self.configs.task_name == "imputation":
            assert self.seq_len == self.pred_len, f"--seq_len {self.seq_len} must be equal to --pred_len {self.pred_len} for imputation!"
            assert self.configs.missing_rate > 0, f"--missing_rate {self.configs.missing_rate} should be greater than 0 for imputation!"
            assert self.seq_len <= self.L_TOTAL and self.pred_len <= self.L_TOTAL, f"Either {self.seq_len=} or {self.pred_len=} is too large. Expect their values smaller than self.L_TOTAL"
        elif self.configs.task_name in ["short_term_forecast", "long_term_forecast"]:
            assert self.seq_len + self.pred_len <= self.L_TOTAL, f"{self.seq_len+self.pred_len=} is too large. Expect the value smaller than self.L_TOTAL"
        else:
            raise NotImplementedError()

    def _preprocess(self):
        if self.configs.task_name == "imputation":
            backbone_pred_len = 300 # data in pred_len unused. not set to 0 to prevent error
        elif self.configs.task_name in ["short_term_forecast", "long_term_forecast"]:
            backbone_pred_len = self.pred_len
        else:
            raise NotImplementedError()

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

            if self.configs.collate_fn == "collate_fn_fractal":
                '''
                Determine `seq_residual_len`: the maximum length of split position for lookback window and forecast window first.
                '''
                PATCH_LEN = self.configs.patch_len_list[-1] # get the smallest patch length
                n_patch: int = SEQ_LEN // PATCH_LEN
                for sample in test_all_data:
                    x_y = torch.cat([sample['x'], sample['y']], dim=0)
                    x_y_mark = torch.cat([sample["x_mark"], sample["y_mark"]], dim=0)

                    observations_left_bound = x_y_mark < (n_patch * PATCH_LEN / (SEQ_LEN + PRED_LEN))
                    observations_right_bound = x_y_mark < (SEQ_LEN / (SEQ_LEN + PRED_LEN))
                    sample_mask = slice(observations_left_bound.sum(), observations_right_bound.sum())
                    x_y_seq_residual = x_y[sample_mask]
                    seq_residual_len_current = len(x_y_seq_residual)
                    if seq_residual_len_current > seq_residual_len:
                        seq_residual_len = seq_residual_len_current
            else:
                PATCH_LEN = self.configs.patch_len

            for sample in test_all_data:
                if sample["x"].shape[0] > self.seq_len_max_irr:
                    self.seq_len_max_irr = sample["x"].shape[0]
                if sample["y"].shape[0] > self.pred_len_max_irr:
                    self.pred_len_max_irr = sample["y"].shape[0]

                if self.configs.collate_fn in ["collate_fn_patch", "collate_fn_tpatch"]:
                    n_patch: int = math.ceil(SEQ_LEN / PATCH_LEN)
                    n_patch_y: int = math.ceil(PRED_LEN / PATCH_LEN)

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
                        observations = sample["y_mark"] < ((SEQ_LEN + (j + 1) * PATCH_LEN) / (SEQ_LEN + PRED_LEN))
                        patch_j_end = observations.sum()
                        sample_mask = slice(patch_j_end_previous, patch_j_end)
                        y_patch_j = sample["y"][sample_mask]
                        if len(y_patch_j) > self.patch_len_max_irr:
                            self.patch_len_max_irr = len(y_patch_j)

                        patch_j_end_previous = patch_j_end
                elif self.configs.collate_fn == "collate_fn_fractal":
                    '''
                    fractal use the entire time series to split patch, which does not require seq_len % patch_len == 0

                    The goal here is to pad the sequence into a shape, that not only can be divisible by n_patch_all, but also can be splitted directly into lookback and forecast window along time dimension.
                    '''
                    if self.configs.task_name == "imputation":
                        TOTAL_LEN = SEQ_LEN
                    elif self.configs.task_name in ["short_term_forecast", "long_term_forecast"]:
                        TOTAL_LEN = SEQ_LEN + PRED_LEN
                    else:
                        raise NotImplementedError()
                    n_patch_all: int = math.ceil(TOTAL_LEN / PATCH_LEN)
                    x_y = torch.cat([sample['x'], sample['y']], dim=0)
                    x_y_mark = torch.cat([sample["x_mark"], sample["y_mark"]], dim=0)

                    # determine the split position of seq_len and pred_len for current sample
                    n_patch: int = SEQ_LEN // PATCH_LEN
                    observations_left_bound = x_y_mark < (n_patch * PATCH_LEN / (SEQ_LEN + PRED_LEN))
                    observations_right_bound = x_y_mark < (SEQ_LEN / (SEQ_LEN + PRED_LEN))
                    sample_mask = slice(observations_left_bound.sum(), observations_right_bound.sum())
                    x_y_seq_residual = x_y[sample_mask]
                    seq_residual_len_current = len(x_y_seq_residual)

                    patch_i_end_previous = 0
                    for i in range(n_patch_all):
                        observations = x_y_mark < ((i + 1) * PATCH_LEN / (SEQ_LEN + PRED_LEN))
                        patch_i_end = observations.sum()
                        sample_mask = slice(patch_i_end_previous, patch_i_end)
                        x_y_patch_i = x_y[sample_mask]
                        if i == n_patch:
                            # split position in current patch
                            if seq_residual_len + (len(x_y_patch_i) - seq_residual_len_current) > self.patch_len_max_irr:
                                self.patch_len_max_irr = seq_residual_len + (len(x_y_patch_i) - seq_residual_len_current)
                        else:
                            if len(x_y_patch_i) > self.patch_len_max_irr:
                                self.patch_len_max_irr = len(x_y_patch_i)

                        patch_i_end_previous = patch_i_end

            if self.configs.collate_fn in ["collate_fn_patch", "collate_fn_tpatch"]:
                n_patch: int = math.ceil(SEQ_LEN / PATCH_LEN)
                n_patch_y: int = math.ceil(PRED_LEN / PATCH_LEN)
                self.seq_len_max_irr = max(self.seq_len_max_irr, self.patch_len_max_irr * n_patch)
                self.pred_len_max_irr = max(self.pred_len_max_irr, self.patch_len_max_irr * n_patch_y)
            elif self.configs.collate_fn == "collate_fn_fractal":
                n_patch: int = SEQ_LEN // PATCH_LEN
                n_patch_all: int = math.ceil(TOTAL_LEN / PATCH_LEN)
                TOTAL_LEN_MAX_IRR = n_patch_all * self.patch_len_max_irr
                self.seq_len_max_irr = max(self.seq_len_max_irr, n_patch * self.patch_len_max_irr + seq_residual_len)
                self.pred_len_max_irr = TOTAL_LEN_MAX_IRR - self.seq_len_max_irr

                # calculate number of patch at each fractal level
                n_patch_all_list: list[int] = []
                previous_n_patch_all: int = math.ceil((SEQ_LEN + PRED_LEN) / PATCH_LEN) # last level
                for i in range(len(self.configs.patch_len_list)):
                    if i == 0:
                        n_patch_all_list.append(previous_n_patch_all)
                    else:
                        current_n_patch_all = math.ceil(previous_n_patch_all / (self.configs.patch_len_list[i - 1] // self.configs.patch_len_list[i]))
                        n_patch_all_list.append(current_n_patch_all)
                        previous_n_patch_all = current_n_patch_all
                n_patch_all_list.reverse() # -> first level to last level

                # get the max length (real time) among all levels
                length_list = [n_patch * current_length for n_patch, current_length in zip(n_patch_all_list, self.configs.patch_len_list)]
                TOTAL_LEN_MAX = max(length_list)
                # convert to number of smallest patches needed to append at last
                n_patch_pad = (TOTAL_LEN_MAX - length_list[-1]) / PATCH_LEN
                self.pred_len_max_irr += int(n_patch_pad * self.patch_len_max_irr)

            if self.configs.task_name == "imputation":
                # force overwrite pred_len_max_irr for imputation
                self.pred_len_max_irr = self.seq_len_max_irr

            # create a new field in global configs to pass information to models
            self.configs.seq_len_max_irr = self.seq_len_max_irr
            self.configs.pred_len_max_irr = self.pred_len_max_irr
            if self.configs.collate_fn in ["collate_fn_patch", "collate_fn_tpatch", "collate_fn_fractal"]:
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