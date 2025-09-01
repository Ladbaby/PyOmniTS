# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch import Tensor
from einops import *

from utils.ExpConfigs import ExpConfigs
from utils.globals import logger

class Model(nn.Module):
    """
    - Paper: "Recurrent Neural Networks for Multivariate Time Series with Missing Values" (Scientific Reports 2018)
    - Paper link: https://www.nature.com/articles/s41598-018-24271-9
    - Code adapted from: https://github.com/WenjieDu/PyPOTS

    The core wrapper assembles the submodules of GRU-D imputation model
    and takes over the forward progress of the algorithm.
    """
    def __init__(
        self,
        configs: ExpConfigs,
    ):
        super().__init__()
        # BEGIN adaptor
        self.pred_len = configs.pred_len_max_irr or configs.pred_len
        if configs.task_name in ["short_term_forecast", "long_term_forecast"]:
            if configs.seq_len_max_irr is not None:
                n_steps = configs.seq_len_max_irr + configs.pred_len_max_irr
            else: # others
                n_steps = configs.seq_len + configs.pred_len
        elif configs.task_name in ["imputation", "classification"]:
            n_steps = configs.seq_len_max_irr or configs.seq_len
        else:
            raise NotImplementedError
        n_features = configs.enc_in
        rnn_hidden_size = configs.d_model # 100
        # END adaptor

        self.configs = configs
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size

        # create models
        self.backbone = BackboneGRUD(
            n_steps,
            n_features,
            rnn_hidden_size,
        )
        self.output_projection = nn.Linear(rnn_hidden_size, n_features)
        self.decoder_classification = nn.Linear(rnn_hidden_size, configs.n_classes)

    def convert_to_delta(self, x_mark: torch.Tensor) -> torch.Tensor:
        # Ensure x_mark is of shape [batch_size, n_steps, n_features]
        batch_size, n_steps, n_features = x_mark.shape
        
        # Initialize delta tensor with the same shape as x_mark
        delta = torch.zeros((batch_size, n_steps, n_features), device=x_mark.device)
        
        # Calculate deltas for each feature
        # Use slicing to get the difference between current and previous timesteps
        delta[:, 1:, :] = x_mark[:, 1:, :] - x_mark[:, :-1, :]
        
        # The first time step can be set to zero or left as is
        # delta[:, 0, :] = 0  # Optional, already initialized to zero

        return delta

    def calculate_empirical_mean(self, x: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
        # Ensure x is of shape [batch_size, n_steps, n_features]
        # Ensure missing_mask is of the same shape and consists of binary values (0 or 1)
        
        # Invert the missing mask to get valid (observed) values
        valid_mask = 1 - missing_mask  # 1 for observed values, 0 for missing values
        
        # Multiply x by the valid mask to zero out missing values
        x_valid = x * valid_mask
        
        # Count the number of valid (observed) entries for each feature
        count_valid = valid_mask.sum(dim=1)  # Shape: [batch_size, n_features]
        
        # Calculate empirical mean, avoiding division by zero
        empirical_mean = x_valid.sum(dim=1) / count_valid
        empirical_mean[count_valid == 0] = 0  # Handle cases where no valid values exist

        return empirical_mean

    def fill_locf(self, x: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
        # Ensure x is of shape [batch_size, n_steps, n_features]
        batch_size, n_steps, n_features = x.shape
        
        # Create a tensor to store the indices of the last observed values
        last_observed_indices = torch.zeros((batch_size, n_features), dtype=torch.long, device=x.device)
        
        # Create a mask for valid (non-missing) values
        valid_mask = ~missing_mask.bool()
        
        # Use torch.arange to create a tensor of indices
        indices = torch.arange(n_steps, device=x.device).unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, n_features)
        
        # Update last_observed_indices with the most recent valid index for each feature
        last_observed_indices = torch.where(valid_mask, indices, last_observed_indices.unsqueeze(1))
        last_observed_indices, _ = torch.max(last_observed_indices, dim=1)
        
        # Create a tensor of the last observed values
        last_observed_values = x[torch.arange(batch_size).unsqueeze(1), last_observed_indices, torch.arange(n_features)]
        
        # Fill missing values with the last observed values
        return torch.where(missing_mask.bool(), last_observed_values.unsqueeze(1), x)

    def forward(
        self, 
        x: Tensor,
        x_mark: Tensor = None, 
        x_mask: Tensor = None, 
        y: Tensor = None, 
        y_mark: Tensor = None, 
        y_mask: Tensor = None, 
        y_class: Tensor = None, 
        **kwargs
    ) -> dict:
        # BEGIN adaptor
        BATCH_SIZE, SEQ_LEN, ENC_IN = x.shape
        Y_LEN = self.pred_len
        if x_mark is None:
            x_mark = repeat(torch.arange(end=x.shape[1], dtype=x.dtype, device=x.device) / x.shape[1], "L -> B L 1", B=x.shape[0])
        if x_mask is None:
            x_mask = torch.ones_like(x, device=x.device, dtype=x.dtype)
        if y is None:
            if self.configs.task_name in ["short_term_forecast", "long_term_forecast"]:
                logger.warning(f"y is missing for the model input. This is only reasonable when the model is testing flops!")
            y = torch.ones((BATCH_SIZE, Y_LEN, ENC_IN), dtype=x.dtype, device=x.device)
        if y_mark is None:
            y_mark = repeat(torch.arange(end=y.shape[1], dtype=y.dtype, device=y.device) / y.shape[1], "L -> B L 1", B=y.shape[0])
        if y_mask is None:
            y_mask = torch.ones_like(y, device=y.device, dtype=y.dtype)
        if y_class is None:
            if self.configs.task_name == "classification":
                logger.warning(f"y_class is missing for the model input. This is only reasonable when the model is testing flops!")
            y_class = torch.ones((BATCH_SIZE), dtype=x.dtype, device=x.device)

        x_mark = repeat(x_mark[:, :, 0], "b l -> b l f", f=x.shape[-1])
        y_mark = repeat(y_mark[:, :, 0], "b l -> b l f", f=x.shape[-1])
        if self.configs.task_name in ["short_term_forecast", "long_term_forecast"]:
            x_padding = torch.zeros_like(y, device=x.device)
            X = torch.cat([x, x_padding], dim=1)
            missing_mask = torch.cat([x_mask, x_padding], dim=1)
            deltas = self.convert_to_delta(torch.cat([x_mark, y_mark], dim=1))
            empirical_mean = self.calculate_empirical_mean(X, missing_mask)
            X_filledLOCF = self.fill_locf(X, missing_mask)
            # print(f"{X_filledLOCF[0]=}")
            # input()
        elif self.configs.task_name in ["imputation", "classification"]:
            X = x
            missing_mask = x_mask
            deltas = self.convert_to_delta(x_mark)
            empirical_mean = self.calculate_empirical_mean(x, x_mask)
            X_filledLOCF = self.fill_locf(x, x_mask)
        else:
            raise NotImplementedError(f"{self.configs.task_name} not implemented for GRU_D")
        # END adaptor

        representation_collector, hidden_state = self.backbone(X, missing_mask, deltas, empirical_mean, X_filledLOCF) # (BATCH_SIZE, n_steps, rnn_hidden_size), (BATCH_SIZE, rnn_hidden_size)


        if self.configs.task_name in ['long_term_forecast', 'short_term_forecast', "imputation"]:
            # project back the original data space
            reconstruction = self.output_projection(representation_collector)

            imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
            f_dim = -1 if self.configs.features == 'MS' else 0
            PRED_LEN = y.shape[1]
            return {
                "pred": imputed_data[:, -PRED_LEN:, f_dim:],
                "true": y[:, :, f_dim:],
                "mask": y_mask[:, :, f_dim:]
            }
        elif self.configs.task_name == "classification":
            # lengths = self.compute_lengths(x_mark[:, :, 0]).unsqueeze(1) # (BATCH_SIZE, 1)
            # mask = torch.arange(SEQ_LEN)[None, :].to(x.device) >= (lengths[:, None])
            # mask = rearrange(mask, "B 1 L -> B L 1").long()
            # output = torch.sum(representation_collector * (1 - mask), dim=1) / (lengths + 1) # (BATCH_SIZE, rnn_hidden_size)
            output = self.decoder_classification(hidden_state)
            return {
                "pred_class": output,
                "true_class": y_class,
            }
        else:
            raise NotImplementedError

    def compute_lengths(self, timestamps: Tensor):
        '''
        `lengths` of shape (BATCH_SIZE) is the actual time lengths of samples
        '''
        # Create a mask indicating non-zero elements
        mask = timestamps != 0
        # Generate indices for each position in the sequence
        indices = torch.arange(timestamps.size(1), device=timestamps.device).expand_as(timestamps)
        # Compute the maximum index for each row where the element is non-zero
        max_indices = (indices * mask).max(dim=1, keepdim=True)[0]
        # Check if there are any non-zero elements in each row
        any_non_zero = mask.any(dim=1, keepdim=True)
        # Calculate lengths, adjusting for all-zero rows
        lengths = (max_indices + 1) * any_non_zero
        return lengths.squeeze(-1)

class BackboneGRUD(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        rnn_hidden_size: int,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size

        # create models
        self.rnn_cell = nn.GRUCell(self.n_features * 2 + self.rnn_hidden_size, self.rnn_hidden_size)
        self.temp_decay_h = TemporalDecay(input_size=self.n_features, output_size=self.rnn_hidden_size, diag=False)
        self.temp_decay_x = TemporalDecay(input_size=self.n_features, output_size=self.n_features, diag=True)

    def forward(self, X, missing_mask, deltas, empirical_mean, X_filledLOCF) -> Tuple[torch.Tensor, ...]:
        """Forward processing of GRU-D.

        Parameters
        ----------
        X:

        missing_mask:

        deltas:

        empirical_mean:

        X_filledLOCF:

        Returns
        -------
        classification_pred:

        logits:


        """

        hidden_state = torch.zeros((X.size()[0], self.rnn_hidden_size), device=X.device)

        representation_collector = []
        for t in range(self.n_steps):
            # for data, [batch, time, features]
            x = X[:, t, :]  # values
            m = missing_mask[:, t, :]  # mask
            d = deltas[:, t, :]  # delta, time gap
            x_filledLOCF = X_filledLOCF[:, t, :]

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)
            hidden_state = hidden_state * gamma_h
            representation_collector.append(hidden_state)

            x_h = gamma_x * x_filledLOCF + (1 - gamma_x) * empirical_mean
            x_replaced = m * x + (1 - m) * x_h
            data_input = torch.cat([x_replaced, hidden_state, m], dim=1)
            hidden_state = self.rnn_cell(data_input, hidden_state)

        representation_collector = torch.stack(representation_collector, dim=1)

        return representation_collector, hidden_state

class TemporalDecay(nn.Module):
    """The module used to generate the temporal decay factor gamma in the GRU-D model.
    Please refer to the original paper :cite:`che2018GRUD` for more details.

    Attributes
    ----------
    W: tensor,
        The weights (parameters) of the module.
    b: tensor,
        The bias of the module.

    Parameters
    ----------
    input_size : int,
        the feature dimension of the input

    output_size : int,
        the feature dimension of the output

    diag : bool,
        whether to product the weight with an identity matrix before forward processing

    References
    ----------
    .. [1] `Che, Zhengping, Sanjay Purushotham, Kyunghyun Cho, David Sontag, and Yan Liu.
        "Recurrent neural networks for multivariate time series with missing values."
        Scientific reports 8, no. 1 (2018): 6085.
        <https://www.nature.com/articles/s41598-018-24271-9.pdf>`_

    """

    def __init__(self, input_size: int, output_size: int, diag: bool = False):
        super().__init__()
        self.diag = diag
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag:
            assert input_size == output_size
            m = torch.eye(input_size, input_size)
            self.register_buffer("m", m)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        std_dev = 1.0 / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-std_dev, std_dev)
        if self.b is not None:
            self.b.data.uniform_(-std_dev, std_dev)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        """Forward processing of this NN module.

        Parameters
        ----------
        delta : tensor, shape [n_samples, n_steps, n_features]
            The time gaps.

        Returns
        -------
        gamma : tensor, of the same shape with parameter `delta`, values in (0,1]
            The temporal decay factor.
        """
        if self.diag:
            gamma = F.relu(F.linear(delta, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(delta, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma
