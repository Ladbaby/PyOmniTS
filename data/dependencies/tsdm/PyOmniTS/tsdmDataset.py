# Code from: https://github.com/Ladbaby/PyOmniTS
import math
import warnings
from abc import abstractmethod

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, ConcatDataset
from sklearn.model_selection import train_test_split

from utils.globals import logger
from utils.ExpConfigs import ExpConfigs

warnings.filterwarnings('ignore')

class tsdmDataset(Dataset):
    '''
    Base class for MIMIC_III, MIMIC_IV, P12, and USHCN in PyOmniTS
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

        self.cache = True

        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.L_TOTAL = None # set by child class

        self.dataset_root_path = configs.dataset_root_path
        self.sample_index = None # set by child class' _get_sample_index

    def __getitem__(self, index):
        sample = self.dataset[index]
        x_mark, x, y_mark = sample.inputs
        y = sample.targets
        sample_ID = sample.key

        # create a mask for looking up the target values
        y_mask = y.isfinite()
        x_mask = x.isfinite()

        # WARNING: this is not the final input to the model, they should be processed by any collate_fn!
        return {
            "x": x,
            "x_mark": x_mark,
            "x_mask": x_mask,
            "y": y,
            "y_mark": y_mark,
            "y_mask": y_mask,
            "sample_ID": self.sample_index[index],
            "_configs": self.configs, # only used by the collate_fns
            "_L_TOTAL": self.L_TOTAL # only used by the collate_fns
        }

    def __len__(self):
        return len(self.dataset)
    
    def _check_lengths(self):
        if self.configs.task_name in ["short_term_forecast", "long_term_forecast"]:
            assert self.seq_len + self.pred_len <= self.L_TOTAL, f"{self.seq_len+self.pred_len=} is too large. Expect the value smaller than {self.L_TOTAL}"
        else:
            raise NotImplementedError

    def _preprocess_base(self, task):
        """
        preprocess without time alignment
        """
        if self.flag != "val":
            # val set will follow the setting of train set
            # determine the max number of observations along time, among all samples
            dataset = ConcatDataset([
                task.get_dataset((0, "train")), 
                task.get_dataset((0, "val")), 
                task.get_dataset((0, "test")), 
            ]) # use all data to determine max length
            self.seq_len_max_irr = 0
            self.pred_len_max_irr = 0
            self.patch_len_max_irr = 0
            seq_residual_len = 0

            SEQ_LEN = self.configs.seq_len
            PRED_LEN = self.configs.pred_len

            PATCH_LEN = self.configs.patch_len

            for sample in dataset:
                x_mark, x, y_mark = sample.inputs
                y = sample.targets
                if x.shape[0] > self.seq_len_max_irr:
                    self.seq_len_max_irr = x.shape[0]
                if y.shape[0] > self.pred_len_max_irr:
                    self.pred_len_max_irr = y.shape[0]

                if self.configs.collate_fn in ["collate_fn_patch", "collate_fn_tpatch"]:
                    n_patch: int = math.ceil(SEQ_LEN / PATCH_LEN)
                    n_patch_y: int = math.ceil(PRED_LEN / PATCH_LEN)

                    patch_i_end_previous = 0
                    for i in range(n_patch):
                        observations = x_mark < ((i + 1) * PATCH_LEN / self.L_TOTAL)
                        patch_i_end = observations.sum()
                        sample_mask = slice(patch_i_end_previous, patch_i_end)
                        x_patch_i = x[sample_mask]
                        if len(x_patch_i) > self.patch_len_max_irr:
                            self.patch_len_max_irr = len(x_patch_i)

                        patch_i_end_previous = patch_i_end

                    patch_j_end_previous = 0
                    for j in range(n_patch_y):
                        observations = y_mark < ((SEQ_LEN + (j + 1) * PATCH_LEN) / self.L_TOTAL)
                        patch_j_end = observations.sum()
                        sample_mask = slice(patch_j_end_previous, patch_j_end)
                        y_patch_j = y[sample_mask]
                        if len(y_patch_j) > self.patch_len_max_irr:
                            self.patch_len_max_irr = len(y_patch_j)

                        patch_j_end_previous = patch_j_end

            if self.configs.collate_fn in ["collate_fn_patch", "collate_fn_tpatch"]:
                n_patch: int = math.ceil(SEQ_LEN / PATCH_LEN)
                n_patch_y: int = math.ceil(PRED_LEN / PATCH_LEN)
                self.seq_len_max_irr = max(self.seq_len_max_irr, self.patch_len_max_irr * n_patch)
                self.pred_len_max_irr = max(self.pred_len_max_irr, self.patch_len_max_irr * n_patch_y)
            
            del dataset

            # create a new field in global configs to pass information to models
            self.configs.seq_len_max_irr = self.seq_len_max_irr
            self.configs.pred_len_max_irr = self.pred_len_max_irr
            # self.configs.pred_len_max_irr = max(self.pred_len_max_irr, self.patch_len_max_irr * n_patch_y)
            if self.configs.collate_fn in ["collate_fn_patch", "collate_fn_tpatch"]:
                self.configs.patch_len_max_irr = self.patch_len_max_irr
                logger.debug(f"{self.configs.patch_len_max_irr=}")
            logger.debug(f"{self.configs.seq_len_max_irr=}")
            logger.debug(f"{self.configs.pred_len_max_irr=}")


        if self.flag == "test_all":
            # merge the 3 datasets
            dataset = ConcatDataset([
                task.get_dataset((0, "train")), 
                task.get_dataset((0, "val")), 
                task.get_dataset((0, "test")), 
            ])
        else:
            dataset = task.get_dataset(
                (0, self.flag)
            )

        self.dataset = dataset

    @abstractmethod
    def _preprocess(self):
        ...

    @abstractmethod
    def _get_sample_index(self):
        ...
    
def fix_nan_x_mark(x_mark, seq_len, l_total):
    # Create a tensor of indices
    BATCH_SIZE, SEQ_LEN_MAX_IRR, _ = x_mark.shape
    indices = torch.linspace(start=seq_len / l_total - 2 * 0.01, end=seq_len / l_total - 0.001, steps=SEQ_LEN_MAX_IRR).to(x_mark.device).view(1, -1, 1).repeat(BATCH_SIZE, 1, 1)

    # Create a mask for NaN values
    nan_mask = torch.isnan(x_mark)

    # Fill NaN values using the mask
    x_mark[nan_mask] = indices[nan_mask]

    return x_mark

def fix_nan_y_mark(y_mark):
    # Create a tensor of indices
    BATCH_SIZE, PRED_LEN, _ = y_mark.shape
    indices = torch.linspace(start=1 - 2 * 0.01, end=1 - 0.001, steps=PRED_LEN).to(y_mark.device).view(1, -1, 1).repeat(BATCH_SIZE, 1, 1)

    # Create a mask for NaN values
    nan_mask = torch.isnan(y_mark)

    # Fill NaN values using the mask
    y_mark[nan_mask] = indices[nan_mask]

    return y_mark

def collate_fn(
    batch: list[dict[str, Tensor|ExpConfigs]],
) -> dict[str, Tensor]:
    '''
    rewrite the collate_fn to return dictionary of Tensors, aligning with api

    returns:
    - x, x_mask: [BATCH_SIZE, SEQ_LEN_MAX_IRR, ENC_IN]
    - x_mark: [BATCH_SIZE, SEQ_LEN_MAX_IRR, 1]
    - y, y_mask: [BATCH_SIZE, PRED_LEN_MAX_IRR, ENC_IN]
    - y_mark: [BATCH_SIZE, PRED_LEN_MAX_IRR, 1]
    - sample_ID: [BATCH_SIZE]
    '''
    configs = batch[0]["_configs"]
    L_TOTAL = batch[0]["_L_TOTAL"]
    seq_len_max_irr: int = configs.seq_len_max_irr
    pred_len_max_irr: int = configs.pred_len_max_irr

    xs: list[Tensor] = []
    ys: list[Tensor] = []
    x_marks: list[Tensor] = []
    y_marks: list[Tensor] = []
    x_masks: list[Tensor] = []
    y_masks: list[Tensor] = []
    sample_IDs: list[int] = []

    for sample_dict in batch:
        x = sample_dict['x']
        x_mark = sample_dict['x_mark']
        x_mask = sample_dict['x_mask']
        y = sample_dict['y']
        y_mark = sample_dict['y_mark']
        y_mask = sample_dict['y_mask']
        sample_ID = sample_dict["sample_ID"]

        xs.append(x)
        x_marks.append(x_mark)
        x_masks.append(x_mask)

        ys.append(y)
        y_marks.append(y_mark)
        y_masks.append(y_mask)

        sample_IDs.append(sample_ID)

    ENC_IN = xs[0].shape[-1]

    # to ensure padding to n_observations_max, we manually append a sample with desired shape then removed.
    xs.append(torch.zeros(seq_len_max_irr, ENC_IN))
    x_marks.append(torch.zeros(seq_len_max_irr))
    x_masks.append(torch.zeros(seq_len_max_irr, ENC_IN))
    ys.append(torch.zeros(pred_len_max_irr, ENC_IN))
    y_marks.append(torch.zeros(pred_len_max_irr))
    y_masks.append(torch.zeros(pred_len_max_irr, ENC_IN))

    xs=pad_sequence(xs, batch_first=True, padding_value=float("nan"))
    x_marks=pad_sequence(x_marks, batch_first=True, padding_value=float("nan"))
    x_masks=pad_sequence(x_masks, batch_first=True)
    ys=pad_sequence(ys, batch_first=True, padding_value=float("nan"))
    y_marks=pad_sequence(y_marks, batch_first=True, padding_value=float("nan"))
    y_masks=pad_sequence(y_masks, batch_first=True)

    xs = xs[:-1]
    x_marks = x_marks[:-1]
    x_masks = x_masks[:-1]
    if configs.task_name in ["short_term_forecast", "long_term_forecast"]:
        ys = ys[:-1]
        y_marks = y_marks[:-1]
        y_masks = y_masks[:-1]
    elif configs.task_name == "imputation":
        ys = xs.clone()
        y_marks = x_marks.clone()
        y_masks = x_masks.clone()
    else:
        raise NotImplementedError

    sample_IDs = torch.tensor(sample_IDs).float()

    if configs.missing_rate > 0:
        # manually mask out some observations in input
        # Flatten the mask and data tensor
        flat_mask = x_masks.view(-1)
        flat_x = xs.view(-1)

        # Find indices of available data (where mask is 1)
        available_flat_indices = torch.where(flat_mask == 1)[0]
        num_available = available_flat_indices.size(0)
        num_to_mask = int(configs.missing_rate * num_available)

        if num_to_mask > 0:
            # Generate random permutation on the same device
            perm = torch.randperm(num_available, device=available_flat_indices.device)
            selected_flat = available_flat_indices[perm[:num_to_mask]]
            
            # Apply masking to x and x_mask. In-place operation
            flat_x[selected_flat] = torch.nan
            flat_mask[selected_flat] = 0
        else:
            logger.warning(f"Number of observations {num_available} * missing rate {configs.missing_rate} = {num_to_mask} observations to be masked. Tips: either observations are too sparse, or --missing_rate is too small. Consider increase --missing_rate.")

        if configs.task_name == "imputation":
            y_masks = y_masks.int() - x_masks.int()
            ys = ys - xs


    return {
        "x": torch.nan_to_num(xs),
        "x_mark": fix_nan_x_mark(x_marks.unsqueeze(-1), seq_len=configs.seq_len, l_total=L_TOTAL).float(),
        "x_mask": x_masks.float(),
        "y": torch.nan_to_num(ys),
        "y_mark": fix_nan_y_mark(y_marks.unsqueeze(-1)).float(),
        "y_mask": y_masks.float(),
        "sample_ID": sample_IDs
    }

def collate_fn_patch(
    batch: list[dict[str, Tensor|ExpConfigs]],
) -> dict[str, Tensor]:
    '''
    patchify version of collate_fn

    returns:
    - x, x_mask: [BATCH_SIZE, PATCH_LEN_MAX_IRR * N_PATCH, ENC_IN]
    - x_mark: [BATCH_SIZE, PATCH_LEN_MAX_IRR * N_PATCH, 1]
    - y, y_mask: [BATCH_SIZE, PATCH_LEN_MAX_IRR * N_PATCH_Y, ENC_IN]
    - y_mark: [BATCH_SIZE, PATCH_LEN_MAX_IRR * N_PATCH_Y, 1]
    - sample_ID: [BATCH_SIZE]
    '''
    configs = batch[0]["_configs"]
    L_TOTAL = batch[0]["_L_TOTAL"]
    seq_len_max_irr: int = configs.seq_len_max_irr
    pred_len_max_irr: int = configs.pred_len_max_irr
    # actual patch length can be smaller or even greater than configs.patch_len, depending on the actual sampling rate of the irregular time series
    # because configs.patch_len is describing number of time units (e.g., 12 hours), but patch_len_max_irr is describing number of actual observations
    patch_len_max_irr: int = configs.patch_len_max_irr

    xs: list[Tensor] = []
    ys: list[Tensor] = []
    x_marks: list[Tensor] = []
    y_marks: list[Tensor] = []
    x_masks: list[Tensor] = []
    y_masks: list[Tensor] = []
    sample_IDs: list[int] = []

    PATCH_LEN = configs.patch_len
    SEQ_LEN = configs.seq_len
    PRED_LEN = configs.pred_len
    n_patch: int = math.ceil(SEQ_LEN / PATCH_LEN)
    n_patch_y: int = math.ceil(PRED_LEN / PATCH_LEN)

    for sample_dict in batch:
        x = sample_dict['x']
        x_mark = sample_dict['x_mark']
        x_mask = sample_dict['x_mask']
        y = sample_dict['y']
        y_mark = sample_dict['y_mark']
        y_mask = sample_dict['y_mask']
        sample_ID = sample_dict["sample_ID"]

        patch_i_end_previous = 0

        for i in range(n_patch):
            observations = x_mark < ((i + 1) * PATCH_LEN / L_TOTAL)
            patch_i_end = observations.sum()
            sample_mask = slice(patch_i_end_previous, patch_i_end)
            x_patch_i = x[sample_mask]
            if len(x_patch_i) == 0:
                xs.append(torch.full((1, x.shape[-1]), fill_value=float("nan"), device=x.device))
                x_marks.append(torch.zeros((1), device=x.device))
                x_masks.append(torch.zeros((1, x.shape[-1]), device=x.device))
            else:
                xs.append(x_patch_i)
                x_marks.append(x_mark[sample_mask])
                x_masks.append(x_mask[sample_mask])

            patch_i_end_previous = patch_i_end

        patch_j_end_previous = 0

        for j in range(n_patch_y):
            observations = y_mark < ((SEQ_LEN + (j + 1) * PATCH_LEN) / L_TOTAL)
            patch_j_end = observations.sum()
            sample_mask = slice(patch_j_end_previous, patch_j_end)
            y_patch_j = y[sample_mask]
            if len(y_patch_j) == 0:
                ys.append(torch.full((1, y.shape[-1]), fill_value=float("nan"), device=y.device))
                y_marks.append(torch.zeros((1), device=y.device))
                y_masks.append(torch.zeros((1, y.shape[-1]), device=y.device))
            else:
                ys.append(y_patch_j)
                y_marks.append(y_mark[sample_mask])
                y_masks.append(y_mask[sample_mask])

            patch_j_end_previous = patch_j_end

        sample_IDs.append(sample_ID)

    ENC_IN = xs[0].shape[-1]

    # manually append a sample with desired shape then removed.
    xs.append(torch.zeros(patch_len_max_irr, ENC_IN))
    x_marks.append(torch.zeros(patch_len_max_irr))
    x_masks.append(torch.zeros(patch_len_max_irr, ENC_IN))
    ys.append(torch.zeros(patch_len_max_irr, ENC_IN))
    y_marks.append(torch.zeros(patch_len_max_irr))
    y_masks.append(torch.zeros(patch_len_max_irr, ENC_IN))

    xs=pad_sequence(xs, batch_first=True, padding_value=float("nan"))
    x_marks=pad_sequence(x_marks, batch_first=True)
    x_masks=pad_sequence(x_masks, batch_first=True)
    ys=pad_sequence(ys, batch_first=True, padding_value=float("nan"))
    y_marks=pad_sequence(y_marks, batch_first=True)
    y_masks=pad_sequence(y_masks, batch_first=True)

    xs = xs[:-1]
    x_marks = x_marks[:-1]
    x_masks = x_masks[:-1]
    if configs.task_name in ["short_term_forecast", "long_term_forecast"]:
        ys = ys[:-1]
        y_marks = y_marks[:-1]
        y_masks = y_masks[:-1]
    elif configs.task_name == "imputation":
        ys = xs.clone()
        y_marks = x_marks.clone()
        y_masks = x_masks.clone()
    else:
        raise NotImplementedError

    sample_IDs = torch.tensor(sample_IDs).float()

    if configs.missing_rate > 0:
        # manually mask out some observations in input
        # Flatten the mask and data tensor
        flat_mask = x_masks.view(-1)
        flat_x = xs.view(-1)

        # Find indices of available data (where mask is 1)
        available_flat_indices = torch.where(flat_mask == 1)[0]
        num_available = available_flat_indices.size(0)
        num_to_mask = int(configs.missing_rate * num_available)

        if num_to_mask > 0:
            # Generate random permutation on the same device
            perm = torch.randperm(num_available, device=available_flat_indices.device)
            selected_flat = available_flat_indices[perm[:num_to_mask]]
            
            # Apply masking to x and x_mask. In-place operation
            flat_x[selected_flat] = torch.nan
            flat_mask[selected_flat] = 0
        else:
            logger.warning(f"Number of observations {num_available} * missing rate {configs.missing_rate} = {num_to_mask} observations to be masked. Tips: either observations are too sparse, or --missing_rate is too small. Consider increase --missing_rate.")

        if configs.task_name == "imputation":
            y_masks = y_masks.int() - x_masks.int()
            ys = ys - xs

    # note that patch_len_max_irr * n_patch does not necessarily equal to configs.seq_len. see patch_len_max_irr definition for explanation
    return {
        "x": torch.nan_to_num(xs.view(-1, patch_len_max_irr * n_patch, ENC_IN)),
        "x_mark": x_marks.view(-1, patch_len_max_irr * n_patch).unsqueeze(-1).float(),
        "x_mask": x_masks.view(-1, patch_len_max_irr * n_patch, ENC_IN).float(),
        "y": torch.nan_to_num(ys.view(-1, patch_len_max_irr * n_patch_y, ENC_IN)),
        "y_mark": y_marks.view(-1, patch_len_max_irr * n_patch_y).unsqueeze(-1).float(),
        "y_mask": y_masks.view(-1, patch_len_max_irr * n_patch_y, ENC_IN).float(),
        "sample_ID": sample_IDs
    }

def collate_fn_tpatch(
    batch: list[dict[str, Tensor|ExpConfigs]],
) -> dict[str, Tensor]:
    '''
    tPatchGNN approach.
    WARNING: can only be used with --is_training 0 and --test_dataset_statistics 1 to analyze the number of observations after padding!!! 
    Since the returned tensor shapes are not aligned with PyOmniTS's API, it cannot be used for training.
    '''
    configs = batch[0]["_configs"]
    L_TOTAL = batch[0]["_L_TOTAL"]

    xs: list[Tensor] = []
    ys: list[Tensor] = []
    x_marks: list[Tensor] = []
    y_marks: list[Tensor] = []
    x_masks: list[Tensor] = []
    y_masks: list[Tensor] = []
    sample_IDs: list[int] = []

    PATCH_LEN = configs.patch_len
    SEQ_LEN = configs.seq_len
    PRED_LEN = configs.pred_len
    n_patch: int = math.ceil(SEQ_LEN / PATCH_LEN)
    n_patch_y: int = math.ceil(PRED_LEN / PATCH_LEN)

    for sample_dict in batch:
        x = sample_dict['x']
        x_mark = sample_dict['x_mark']
        x_mask = sample_dict['x_mask']
        y = sample_dict['y']
        y_mark = sample_dict['y_mark']
        y_mask = sample_dict['y_mask']
        sample_ID = sample_dict["sample_ID"]

        patch_i_end_previous = 0

        for i in range(n_patch):
            observations = x_mark < ((i + 1) * PATCH_LEN / L_TOTAL)
            patch_i_end = observations.sum()
            sample_mask = slice(patch_i_end_previous, patch_i_end)
            x_patch_i = x[sample_mask]
            x_mask_patch_i = x_mask[sample_mask]
            for variable in range(x_patch_i.shape[-1]):
                x_patch_i_variable = x_patch_i[:, variable]
                x_mask_patch_i_variable = x_mask_patch_i[:, variable]
                non_zero_mask = x_mask_patch_i_variable > 0
                x_patch_i_non_zero = x_patch_i_variable[non_zero_mask]
                x_mask_patch_i_non_zero = x_mask_patch_i_variable[non_zero_mask]
                if len(x_patch_i_variable) == 0:
                    xs.append(torch.full((1,), fill_value=float("nan"), device=x.device))
                    x_marks.append(torch.zeros((1), device=x.device))
                    x_masks.append(torch.zeros((1), device=x.device))
                else:
                    xs.append(x_patch_i_non_zero)
                    x_marks.append(x_mark[sample_mask][non_zero_mask])
                    x_masks.append(x_mask_patch_i_non_zero)
            
            patch_i_end_previous = patch_i_end

        patch_j_end_previous = 0

        for j in range(n_patch_y):
            observations = y_mark < ((SEQ_LEN + (j + 1) * PATCH_LEN) / L_TOTAL)
            patch_j_end = observations.sum()
            sample_mask = slice(patch_j_end_previous, patch_j_end)
            y_patch_j = y[sample_mask]
            y_mask_patch_j = y_mask[sample_mask]
            for variable in range(y_patch_j.shape[-1]):
                y_patch_j_variable = y_patch_j[:, variable]
                y_mask_patch_j_variable = y_mask_patch_j[:, variable]
                non_zero_mask = y_mask_patch_j_variable > 0
                y_patch_j_non_zero = y_patch_j_variable[non_zero_mask]
                y_mask_patch_j_non_zero = y_mask_patch_j_variable[non_zero_mask]
                if len(y_patch_j_variable) == 0:
                    ys.append(torch.full((1,), fill_value=float("nan"), device=y.device))
                    y_marks.append(torch.zeros((1), device=y.device))
                    y_masks.append(torch.zeros((1), device=y.device))
                else:
                    ys.append(y_patch_j_non_zero)
                    y_marks.append(y_mark[sample_mask][non_zero_mask])
                    y_masks.append(y_mask_patch_j_non_zero)
            
            patch_j_end_previous = patch_j_end

        sample_IDs.append(sample_ID)

    xs=pad_sequence(xs, batch_first=True, padding_value=float("nan"))
    x_marks=pad_sequence(x_marks, batch_first=True)
    x_masks=pad_sequence(x_masks, batch_first=True)
    ys=pad_sequence(ys, batch_first=True, padding_value=float("nan"))
    y_marks=pad_sequence(y_marks, batch_first=True)
    y_masks=pad_sequence(y_masks, batch_first=True)

    sample_IDs = torch.tensor(sample_IDs).float()

    if configs.missing_rate > 0:
        # manually mask out some observations in input
        # Flatten the mask and data tensor
        flat_mask = x_masks.view(-1)
        flat_x = xs.view(-1)

        # Find indices of available data (where mask is 1)
        available_flat_indices = torch.where(flat_mask == 1)[0]
        num_available = available_flat_indices.size(0)
        num_to_mask = int(configs.missing_rate * num_available)

        if num_to_mask > 0:
            # Generate random permutation on the same device
            perm = torch.randperm(num_available, device=available_flat_indices.device)
            selected_flat = available_flat_indices[perm[:num_to_mask]]
            
            # Apply masking to x and x_mask. In-place operation
            flat_x[selected_flat] = torch.nan
            flat_mask[selected_flat] = 0
        else:
            logger.warning(f"Number of observations {num_available} * missing rate {configs.missing_rate} = {num_to_mask} observations to be masked. Tips: either observations are too sparse, or --missing_rate is too small. Consider increase --missing_rate.")

        if configs.task_name == "imputation":
            y_masks = y_masks.int() - x_masks.int()
            ys = ys - xs

    return {
        "x": torch.nan_to_num(xs),
        "x_mark": x_marks.unsqueeze(-1).float(),
        "x_mask": x_masks.float(),
        "y": torch.nan_to_num(ys),
        "y_mark": y_marks.unsqueeze(-1).float(),
        "y_mask": y_masks.float(),
        "sample_ID": sample_IDs
    }