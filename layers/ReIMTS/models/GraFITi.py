# Code from: https://github.com/Ladbaby/PyOmniTS
import torch
import torch.nn as nn
from einops import *
from torch import Tensor

from layers.ReIMTS.layers.GraFITi import GraFITi_layers
from utils.ExpConfigs import ExpConfigs
from utils.globals import logger


class Model(nn.Module):
    '''
    - paper: "GraFITi: Graphs for Forecasting Irregularly Sampled Time Series" (AAAI 2024)
    - paper link: https://ojs.aaai.org/index.php/AAAI/article/view/29560
    - code adapted from: https://github.com/yalavarthivk/GraFITi
    '''
    def __init__(
        self,
        configs: ExpConfigs,
        **kwargs
    ):
        super().__init__()
        self.configs = configs
        self.dim=configs.enc_in
        self.pred_len = configs.pred_len_max_irr or configs.pred_len
        self.attn_head = configs.n_heads # 4
        self.latent_dim = configs.d_model # 128
        self.n_layers = configs.n_layers # 2
        self.enc = GraFITi_layers.Encoder(self.dim, self.latent_dim, self.n_layers, self.attn_head, self.configs.task_name, self.configs.n_classes)

    def get_extrapolation(self, context_x, context_w, target_x, target_y, x_repr_var, exp_stage):
        context_mask = context_w[:, :, self.dim:]
        X = context_w[:, :, :self.dim]
        X = X*context_mask
        context_mask = context_mask + target_y[:,:,self.dim:]
        return self.enc(context_x, X, context_mask, target_y[:,:,:self.dim], target_y[:,:,self.dim:], x_repr_var, exp_stage)

    def convert_data(self,  x_time, x_vals, x_mask, y_time, y_vals, y_mask):
        return x_time, torch.cat([x_vals, x_mask],-1), y_time, torch.cat([y_vals, y_mask],-1)  

    def forward(
        self, 
        x: Tensor,
        x_mark: Tensor = None, 
        x_mask: Tensor = None, 
        x_repr_var: Tensor = None,
        y: Tensor = None, 
        y_mark: Tensor = None, 
        y_mask: Tensor = None,
        y_class: Tensor = None,
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
            if self.configs.task_name in ["short_term_forecast", "long_term_forecast", "imputation"]:
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

        x_mark = x_mark[:, :, 0]
        y_mark = y_mark[:, :, 0]

        if self.configs.task_name in ["short_term_forecast", "long_term_forecast", "classification", "representation_learning"]:
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
            raise NotImplementedError()
        # END adaptor


        context_x, context_y, target_x, target_y = self.convert_data(x_mark_new, x_new, x_mask_new, y_mark_new, y_new, y_mask_new)
        if len(context_y.shape) == 2:
            context_x = context_x.unsqueeze(0)
            context_y = context_y.unsqueeze(0)
            target_x = target_x.unsqueeze(0)
            target_y = target_y.unsqueeze(0)

        output_dict: dict[str, Tensor] = self.get_extrapolation(context_x, context_y, target_x, target_y, x_repr_var, exp_stage)
        if self.configs.task_name in ['long_term_forecast', 'short_term_forecast', "imputation"]:
            output = output_dict["output"].squeeze(-1)
            target_U_ = output_dict["target_U_"]
            target_mask_ = output_dict["target_mask_"]
            channel_embedding = output_dict["channel_embedding"]
            if exp_stage in ["train", "val"]:
                return {
                    "pred": output,
                    "true": target_U_,
                    "mask": target_mask_,
                    "pred_repr_var": channel_embedding
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
                    "mask": y_mask[:, :, f_dim:],
                    "pred_repr_var": channel_embedding
                }
        elif self.configs.task_name == "classification":
            output_class = output_dict["output_class"]
            channel_embedding = output_dict["channel_embedding"]
            return {
                "pred_class": output_class,
                "true_class": y_class,
                "pred_repr_var": channel_embedding
            }
        else:
            raise NotImplementedError()

    # convert the output back to original shape, to align with api
    def unpad_and_reshape(
        self, 
        tensor_flattened: Tensor, 
        original_mask: Tensor, 
        original_shape: tuple
    ):
        original_mask = original_mask.bool()
        device = tensor_flattened.device
        # Initialize the result tensor on the correct device
        result = torch.zeros(original_shape, dtype=tensor_flattened.dtype, device=device)

        # 1. Calculate how many valid elements exist per batch item
        # This replaces len(masked_indices) for every row at once
        # Supports masks of shape (B, L) or (B, H, W)
        counts = original_mask.sum(dim=tuple(range(1, original_mask.dim())))

        # 2. Create a boolean mask for the 'tensor_flattened' source
        # We need to pick the first 'n' elements from each row of tensor_flattened
        batch_size, max_len = tensor_flattened.shape[:2]
        # Creates a grid of indices: [[0,1,2...], [0,1,2...]]
        steps = torch.arange(max_len, device=device).expand(batch_size, max_len)
        src_mask = steps < counts.unsqueeze(-1)

        # 3. Vectorized Assignment
        # result[original_mask] automatically maps to the flattened valid elements
        # tensor_flattened[src_mask] extracts only the unpadded elements
        result[original_mask] = tensor_flattened[src_mask]

        return result