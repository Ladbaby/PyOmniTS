import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from einops import *

from layers.PrimeNet.models import TimeBERTForClassification, TimeBERTForRegression, TimeBERTForInterpolation, TimeBERTConfig, TimeBERTForPretrainingV2, TimeBERTForPretraining
from loss_fns.MSE import Loss as MSE
from loss_fns.CrossEntropyLoss import Loss as CrossEntropyLoss
from utils.ExpConfigs import ExpConfigs
from utils.globals import logger

class Model(nn.Module):
    '''
    - paper: "PrimeNet: Pre-training for Irregular Multivariate Time Series" (AAAI 2023)
    - paper link: https://ojs.aaai.org/index.php/AAAI/article/view/25876
    - code adapted from: https://github.com/ranakroychowdhury/PrimeNet

    Note: the provided model weight in original code can only be used on datasets with enc_in=41, thus default to pretraining task
    '''
    def __init__(
        self, 
        configs: ExpConfigs
    ):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len_max_irr or configs.seq_len
        self.pred_len = configs.pred_len_max_irr or configs.pred_len
        self.patch_len = configs.patch_len_max_irr or configs.patch_len

        self.rec_hidden = configs.d_model # 128
        self.embed_time = 128
        self.classify_pertp = False

        config = TimeBERTConfig(
            dataset=configs.dataset_name,
            input_dim=configs.enc_in,
            cls_query=torch.linspace(0, 1., 128),
            hidden_size=self.rec_hidden,
            embed_time=self.embed_time,
            num_heads=configs.n_heads,
            learn_emb=True,
            freq=configs.freq,
            pooling=configs.primenet_pooling,
            classify_pertp=self.classify_pertp,
            max_length=self.seq_len + self.pred_len,
            dropout=0.3,
            temp=0.05,
            n_classes=configs.n_classes
        )
        self.model = TimeBERTForPretrainingV2(config)
        if configs.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.loss_fn_finetune = MSE(config)
        elif configs.task_name == "classification":
            '''
            used at train_stage == 2, i.e., finetuning
            '''
            # self.model = TimeBERTForClassification(config)
            self.loss_fn_finetune = CrossEntropyLoss(config)
            self.decoder_classification = nn.Sequential(
                nn.Linear(config.hidden_size, 300),
                nn.ReLU(),
                nn.Linear(300, 300),
                nn.ReLU(),
                nn.Linear(300, config.n_classes)
            )
        else:
            raise NotImplementedError

        assert (self.seq_len + self.pred_len) % self.patch_len == 0, f"{self.seq_len+self.pred_len=} should be divisible by {self.patch_len=}"
        self.pred_len = configs.pred_len_max_irr or configs.pred_len
        self.n_patch_all: int = configs.seq_len // configs.patch_len + math.ceil(configs.pred_len / configs.patch_len) # pad pred_len to times of patch_len
        self.patch_len = configs.patch_len_max_irr or configs.patch_len

    def forward(
        self, 
        x: Tensor, 
        x_mark: Tensor = None, 
        x_mask: Tensor = None, 
        y: Tensor = None, 
        y_mark: Tensor = None, 
        y_mask: Tensor = None,
        y_class: Tensor = None,
        train_stage: int = 1,
        **kwargs
    ):
        # BEGIN adaptor
        BATCH_SIZE, SEQ_LEN, ENC_IN = x.shape
        Y_LEN = self.pred_len
        if x_mark is None:
            x_mark = repeat(torch.arange(end=x.shape[1], dtype=x.dtype, device=x.device) / x.shape[1], "L -> B L 1", B=x.shape[0])
        if x_mask is None:
            x_mask = torch.ones_like(x, dtype=x.dtype, device=x.device)
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

        observed_data = torch.cat([x, torch.zeros_like(y).to(y.device)], dim=1).reshape(BATCH_SIZE, self.n_patch_all, self.patch_len, -1)
        observed_tp = torch.cat([x_mark, y_mark], dim=1).reshape(BATCH_SIZE, self.n_patch_all, self.patch_len, -1)[:, :, :, 0]
        observed_mask = torch.cat([x_mask, torch.zeros_like(y_mask).to(y_mask.device)], dim=1).reshape(BATCH_SIZE, self.n_patch_all, self.patch_len, -1)
        interp_mask = torch.cat([torch.zeros_like(x_mask).to(x_mask.device), y_mask], dim=1).reshape(BATCH_SIZE, self.n_patch_all, self.patch_len, -1)
        # END adaptor

        out = self.model(
            torch.cat((observed_data, observed_mask, interp_mask), dim=-1),
            observed_tp
        ) # "loss", "cl_loss", "mse_loss", "correct_num", "total_num", "pred", "cls_pooling", "last_hidden_state"
        if self.configs.task_name in ["short_term_forecast", "long_term_forecast"]:
            pred = rearrange(
                out['pred'], 
                "(B N_PATCH) L N -> B (N_PATCH L) N", 
                B=BATCH_SIZE,
                N_PATCH=self.n_patch_all,
            )
            f_dim = -1 if self.configs.features == 'MS' else 0
            PRED_LEN = y.shape[1]
            if train_stage == 1: # pretraining. Use hybrid loss
                loss = out["loss"]
            elif train_stage == 2: # finetuning. Use MSE
                loss_fn_output = self.loss_fn_finetune(
                    pred=pred[:, -PRED_LEN:, f_dim:],
                    true=y[:, :, f_dim:]
                )
                loss = loss_fn_output["loss"]
            else:
                logger.exception(f"Expect train_stage in [1, 2], got {train_stage=}")
                exit(1)
            return {
                "pred": pred[:, -PRED_LEN:, f_dim:],
                "true": y[:, :, f_dim:],
                "mask": y_mask[:, :, f_dim:],
                "loss": loss
            }
        elif self.configs.task_name == "classification":
            pred = self.decoder_classification(out["cls_pooling"].reshape(BATCH_SIZE, -1, self.rec_hidden).mean(1))
            if train_stage == 1: # pretraining. Use hybrid loss
                loss = out["loss"]
            elif train_stage == 2: # finetuning. Use CrossEntropyLoss
                loss_fn_output = self.loss_fn_finetune(
                    pred_class=pred,
                    true_class=y_class
                )
                loss = loss_fn_output["loss"]
            else:
                logger.exception(f"Expect train_stage in [1, 2], got {train_stage=}")
                exit(1)
            return {
                "pred_class": pred,
                "true_class": y_class,
                "loss": loss
            }
        else:
            raise NotImplementedError