import torch
import torch.nn as nn
from torch import Tensor
from einops import *

from layers.GraFITi import GraFITi_layers
from utils.globals import logger
from utils.ExpConfigs import ExpConfigs

class Model(nn.Module):
    '''
    - paper: "GraFITi: Graphs for Forecasting Irregularly Sampled Time Series" (AAAI 2024)
    - paper link: https://ojs.aaai.org/index.php/AAAI/article/view/29560
    - code adapted from: https://github.com/yalavarthivk/GraFITi
    '''
    def __init__(
        self,
        configs: ExpConfigs
    ):
        super().__init__()
        self.configs = configs
        self.dim=configs.enc_in
        self.pred_len = configs.pred_len_max_irr or configs.pred_len
        self.attn_head = configs.n_heads # 4
        self.latent_dim = configs.d_model # 128
        self.n_layers = configs.n_layers # 2
        self.enc = GraFITi_layers.Encoder(self.dim, self.latent_dim, self.n_layers, self.attn_head, self.configs.task_name, self.configs.n_classes)

    def get_extrapolation(self, context_x, context_w, target_x, target_y, exp_stage):
        context_mask = context_w[:, :, self.dim:]
        X = context_w[:, :, :self.dim]
        X = X*context_mask
        context_mask = context_mask + target_y[:,:,self.dim:]
        if self.configs.task_name in ["short_term_forecast", "long_term_forecast", "imputation"]:
            output, target_U_, target_mask_ = self.enc(context_x, X, context_mask, target_y[:,:,:self.dim], target_y[:,:,self.dim:], exp_stage)
            return output, target_U_, target_mask_
        else:
            raise NotImplementedError

    def convert_data(self,  x_time, x_vals, x_mask, y_time, y_vals, y_mask):
        return x_time, torch.cat([x_vals, x_mask],-1), y_time, torch.cat([y_vals, y_mask],-1)  

    def forward(
        self, 
        x: Tensor,
        x_mark: Tensor = None, 
        x_mask: Tensor = None, 
        y: Tensor = None, 
        y_mark: Tensor = None, 
        y_mask: Tensor = None,
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

        x_mark = x_mark[:, :, 0]
        y_mark = y_mark[:, :, 0]

        if self.configs.task_name in ["short_term_forecast", "long_term_forecast"]:
            x_zero_padding = torch.zeros_like(y, device=x.device)
            y_zero_padding = torch.zeros_like(x, device=y.device)

            x_new = torch.cat([x, x_zero_padding], dim=1)
            original_shape = x_new.shape
            x_mark_new = torch.cat([x_mark, y_mark], dim=1)
            x_mask_new = torch.cat([x_mask, x_zero_padding], dim=1)

            y_new = torch.cat([y_zero_padding, y], dim=1)
            y_mark_new = torch.cat([x_mark, y_mark], dim=1)
            y_mask_new = torch.cat([y_zero_padding, y_mask], dim=1)

            x_y_mask = torch.cat([x_mask, y_mask], dim=1)
        elif self.configs.task_name in ["imputation"]:
            x_new = x
            original_shape = x_new.shape
            x_mark_new = x_mark
            x_mask_new = x_mask

            y_new = y
            y_mark_new = y_mark
            y_mask_new = y_mask

            x_y_mask = x_mask + y_mask
        else:
            raise NotImplementedError
        # END adaptor


        context_x, context_y, target_x, target_y = self.convert_data(x_mark_new, x_new, x_mask_new, y_mark_new, y_new, y_mask_new)
        if len(context_y.shape) == 2:
            context_x = context_x.unsqueeze(0)
            context_y = context_y.unsqueeze(0)
            target_x = target_x.unsqueeze(0)
            target_y = target_y.unsqueeze(0)

        if self.configs.task_name in ['long_term_forecast', 'short_term_forecast', "imputation"]:
            output, target_U_, target_mask_ = self.get_extrapolation(context_x, context_y, target_x, target_y, exp_stage)
            output = output.squeeze(-1)
            if exp_stage in ["train", "val"]:
                return {
                    "pred": output,
                    "true": target_U_,
                    "mask": target_mask_
                }
            else:
                # convert the compressed tensor back to shape [batch_size, seq_len + pred_len, ndims] when testing
                pred = self.unpad_and_reshape(
                    output,
                    x_y_mask,
                    original_shape
                )
                f_dim = -1 if self.configs.features == 'MS' else 0
                PRED_LEN = y.shape[1]
                return {
                    "pred": pred[:, -PRED_LEN:, f_dim:],
                    "true": y[:, :, f_dim:],
                    "mask": y_mask[:, :, f_dim:]
                }
        else:
            raise NotImplementedError

    # convert the output back to original shape, to align with api
    def unpad_and_reshape(self, target_U: Tensor, mask: Tensor, original_shape: Tensor):
        # print(f"{target_U.shape=}")
        # print(f"{mask.shape=}")
        # print(f"{original_shape.shape=}")
        batch_size, time_length, ndims = original_shape
        result = torch.zeros(original_shape, dtype=target_U.dtype, device=target_U.device)

        for i in range(batch_size):
            masked_indices = mask[i].view(-1).nonzero(as_tuple=True)[0]
            unpadded_sequence = target_U[i][:len(masked_indices)]
            result[i].view(-1)[masked_indices] = unpadded_sequence
            
        return result