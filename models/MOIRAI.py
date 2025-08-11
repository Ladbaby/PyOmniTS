import math

import torch
import torch.nn as nn
from torch import Tensor
from einops import *

from layers.uni2ts.model.moirai.module import MoiraiModule
from layers.uni2ts.distribution import MixtureOutput
from layers.uni2ts.distribution import StudentTOutput
from layers.uni2ts.distribution import NormalFixedScaleOutput
from layers.uni2ts.distribution import NegativeBinomialOutput
from layers.uni2ts.distribution import LogNormalOutput
from layers.uni2ts.loss.packed import PackedNLLLoss
from utils.globals import logger
from utils.ExpConfigs import ExpConfigs

class Model(nn.Module):
    '''
    Adaptor class for MOIRAI (uni2ts)

    - paper: "Unified Training of Universal Time Series Forecasting Transformers" (ICML 2024)
    - paper link: https://proceedings.mlr.press/v235/woo24a.html
    - code adapted from: https://github.com/SalesforceAIResearch/uni2ts
    '''
    def __init__(self, configs: ExpConfigs, **kwargs):
        super(Model, self).__init__()
        self.configs = configs

        self.MoiraiModule = MoiraiModule.from_pretrained(
            f"Salesforce/moirai-1.1-R-small",
            resume_download=True,
            distr_output=MixtureOutput(
                components=[
                    StudentTOutput(),
                    NormalFixedScaleOutput(),
                    NegativeBinomialOutput(),
                    LogNormalOutput()
                ]
            ),
            d_model=384,
            num_layers=6,
            patch_sizes=[
                8,
                16,
                32,
                64,
                128
            ],
            max_seq_len=512,
            attn_dropout_p=0.0,
            dropout_p=0.0,
            scaling=True
        )
        self.patch_len_padded = 128 # max value of patch_sizes
    
        self.loss_fn = PackedNLLLoss()

        self.seq_len = configs.seq_len_max_irr or configs.seq_len
        self.pred_len = configs.pred_len_max_irr or configs.pred_len
        self.patch_len = configs.patch_len_max_irr or configs.patch_len

        if self.patch_len > self.patch_len_padded:
            logger.exception(f"Actual patch_len {self.patch_len} is longer than MOIRAI's pretrained max patch length {self.patch_len_padded}. Please specify a smaller --patch_len, or consider retraining MOIRAI with bigger patch size.", stack_info=True)
            exit(1)

        assert (self.seq_len + self.pred_len) % self.patch_len == 0, f"{self.seq_len+self.pred_len=} should be divisible by {self.patch_len=}"
        self.n_patch_all: int = configs.seq_len // configs.patch_len + math.ceil(configs.pred_len / configs.patch_len) # pad pred_len to times of patch_len


    def forward(
        self,
        x: Tensor, 
        x_mark: Tensor = None, 
        x_mask: Tensor = None, 
        y: Tensor = None, 
        y_mark: Tensor = None, 
        y_mask: Tensor = None,
        sample_ID: Tensor = None,
        exp_stage: str = "train",
        **kwargs
    ):
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
        if sample_ID is None:
            sample_ID = torch.arange(end=x.shape[0], dtype=x.dtype, device=x.device)

        # END adaptor

        output, target, prediction_mask, loss = self.forward_all_variate(
            x=x,
            x_mark=x_mark,
            x_mask=x_mask,
            y=y,
            y_mark=y_mark,
            y_mask=y_mask,
            sample_ID=sample_ID,
        )
        if self.configs.task_name in ["long_term_forecast", "short_term_forecast"]:
            if exp_stage == "train":
                return {
                    "pred": self.revert_rearrange_and_pad_multivariate_tensor(output.mean),
                    "true": torch.cat([x, y], dim=1),
                    "mask": torch.cat([x_mask, y_mask], dim=1),
                    "loss": loss
                }
            else:
                f_dim = -1 if self.configs.features == 'MS' else 0
                PRED_LEN = y.shape[1]
                return {
                    "pred": self.revert_rearrange_and_pad_multivariate_tensor(output.mean)[:, -PRED_LEN:, f_dim:],
                    "true": y,
                    "mask": y_mask,
                    "loss": loss
                }
        else:
            raise NotImplementedError

    def forward_one_variate(
        self,
        x: Tensor, 
        x_mark: Tensor, 
        x_mask: Tensor, 
        y: Tensor, 
        y_mark: Tensor, 
        y_mask: Tensor,
        sample_ID: Tensor,
        variate_index: int
    ):
        BATCH_SIZE, SEQ_LEN, ENC_IN = x.shape
        PATCH_LEN = self.configs.patch_len
        PRED_LEN = y.shape[1]
        L = SEQ_LEN + PRED_LEN
        L_NEW = (SEQ_LEN + PRED_LEN) // PATCH_LEN

        x_padding = torch.zeros_like(y)
        y_padding = torch.zeros_like(x)

        sample_id = repeat(
            sample_ID, 
            "BATCH_SIZE -> BATCH_SIZE L_NEW", 
            L_NEW=L_NEW
        ).int()

        observed_mask = self.rearrange_and_pad_multivariate_tensor(torch.cat([x_mask[:, :, variate_index], y_mask[:, :, variate_index]], dim=1)).bool()
        prediction_mask = self.rearrange_and_pad_multivariate_tensor(torch.cat([y_padding[:, :, variate_index], y_mask[:, :, variate_index]], dim=1)).sum(dim=-1).bool()
        variate_id = torch.full((BATCH_SIZE, L_NEW), fill_value=variate_index, device=x.device)
        variate_id = self.rearrange_and_pad_multivariate_tensor(variate_id).sum(dim=-1).int()

        output = self.MoiraiModule(
            target=self.rearrange_and_pad_multivariate_tensor(torch.cat([x[:, :, variate_index], x_padding[:, :, variate_index]], dim=1)),
            observed_mask=observed_mask,
            sample_id=sample_id,
            time_id=self.rearrange_and_pad_multivariate_tensor(torch.cat([x_mark[:, :, 0], y_mark[:, :, 0]], dim=1)).sum(dim=-1).int(),
            variate_id=variate_id,
            prediction_mask=prediction_mask,
            patch_size=torch.full(
                (BATCH_SIZE, L_NEW), 
                fill_value=PATCH_LEN, 
                device=x.device
            )
        )

        target = self.rearrange_and_pad_multivariate_tensor(torch.cat([y_padding[:, :, variate_index], y[:, :, variate_index]], dim=1))

        loss = self.loss_fn(
            pred=output,
            target=target,
            prediction_mask=prediction_mask,
            observed_mask=observed_mask,
            sample_id=sample_id,
            variate_id=variate_id
        )

        return output, target, prediction_mask, loss

    def forward_all_variate(
        self,
        x: Tensor, 
        x_mark: Tensor, 
        x_mask: Tensor, 
        y: Tensor, 
        y_mark: Tensor, 
        y_mask: Tensor,
        sample_ID: Tensor,
    ):
        BATCH_SIZE, SEQ_LEN, ENC_IN = x.shape
        PATCH_LEN = self.patch_len
        PATCH_LEN_MAX = self.patch_len_padded
        PRED_LEN = y.shape[1]
        L = SEQ_LEN + PRED_LEN
        variate_id = repeat(
            torch.arange(ENC_IN, device=x.device),
            "ENC_IN -> BATCH_SIZE L ENC_IN",
            BATCH_SIZE=BATCH_SIZE,
            L=L
        )

        BATCH_SIZE_NEW = BATCH_SIZE * ENC_IN
        L_NEW = self.n_patch_all

        x_padding = torch.zeros_like(y)
        y_padding = torch.zeros_like(x)

        sample_id = repeat(
            sample_ID, 
            "BATCH_SIZE -> (BATCH_SIZE ENC_IN) L_NEW", 
            ENC_IN=ENC_IN, 
            L_NEW=L_NEW
        ).int()

        observed_mask = self.rearrange_and_pad_multivariate_tensor(torch.cat([x_mask, y_mask], dim=1)).bool()
        prediction_mask = self.rearrange_and_pad_multivariate_tensor(torch.cat([y_padding, y_mask], dim=1)).sum(dim=-1).bool()
        variate_id = self.rearrange_and_pad_multivariate_tensor(variate_id).sum(dim=-1).int()

        output = self.MoiraiModule(
            target=self.rearrange_and_pad_multivariate_tensor(torch.cat([x, x_padding], dim=1)),
            observed_mask=observed_mask,
            sample_id=sample_id,
            time_id=self.rearrange_and_pad_multivariate_tensor(
                repeat(
                    torch.cat([x_mark[:, :, 0], y_mark[:, :, 0]], dim=1),
                    "BATCH_SIZE L -> BATCH_SIZE L ENC_IN",
                    ENC_IN=ENC_IN
                )
            ).sum(dim=-1).int(),
            variate_id=variate_id,
            prediction_mask=prediction_mask,
            patch_size=torch.full(
                (BATCH_SIZE_NEW, L_NEW), 
                fill_value=PATCH_LEN, 
                device=x.device
            )
        )

        target = self.rearrange_and_pad_multivariate_tensor(torch.cat([y_padding, y], dim=1))

        loss = self.loss_fn(
            pred=output,
            target=target,
            prediction_mask=prediction_mask,
            observed_mask=observed_mask,
            sample_id=sample_id,
            variate_id=variate_id
        )

        return output, target, prediction_mask, loss

    def rearrange_univariate_tensor(
        self,
        tensor
    ):
        return rearrange(
            tensor,
            "BATCH_SIZE (N_PATCH PATCH_LEN) -> BATCH_SIZE N_PATCH PATCH_LEN",
            BATCH_SIZE=self.configs.batch_size,
            N_PATCH=self.n_patch_all
        )

    def rearrange_and_pad_multivariate_tensor(
        self,
        tensor
    ):
        BATCH_SIZE, L, ENC_IN = tensor.shape
        tensor_rearranged = rearrange(
            tensor,
            "BATCH_SIZE (N_PATCH PATCH_LEN) N -> (BATCH_SIZE N) N_PATCH PATCH_LEN",
            BATCH_SIZE=self.configs.batch_size,
            N_PATCH=self.n_patch_all
        )
        padded_tensor = torch.zeros(BATCH_SIZE * ENC_IN, L // self.patch_len, self.patch_len_padded).to(tensor.device)
        padded_tensor[:, :, :self.patch_len] = tensor_rearranged  # copy original values
        return padded_tensor

    def revert_rearrange_and_pad_multivariate_tensor(
        self,
        tensor
    ):
        return rearrange(
            tensor[:, :, :self.patch_len],
            "(BATCH_SIZE N) N_PATCH PATCH_LEN -> BATCH_SIZE (N_PATCH PATCH_LEN) N",
            BATCH_SIZE=self.configs.batch_size,
        )


