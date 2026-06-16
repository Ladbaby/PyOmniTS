# Code from: https://github.com/Ladbaby/PyOmniTS
import math

import torch
import torch.nn as nn
from torch import Tensor
from einops import repeat

from layers.TFMixer import (
    GlobalRawNUDFT,
    MaskedRevIN,
    PositionalEncoding,
)
from loss_fns.MSE import Loss as MSE
from utils.globals import logger
from utils.ExpConfigs import ExpConfigs


class Model(nn.Module):
    """
    - paper: "Bridging Time and Frequency: A Joint Modeling Framework for Irregular Multivariate Time Series Forecasting" (ICML 2026)
    - paper link: ?
    - code adapted from: https://github.com/decisionintelligence/TFMixer

        Note: This model's code repository originated from PyOmniTS v1.0.0: https://github.com/Ladbaby/PyOmniTS/releases/tag/v1.0.0
        Authors from PyOmniTS have made changes to the code formatting, aligning with the high standard of the PyOmniTS API.
    """
    def __init__(self, configs: ExpConfigs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name

        self.hid_dim = configs.d_model
        self.N = configs.enc_in
        self.dropout = configs.dropout
        self.num_freqs = configs.cru_num_basis
        self.n_patch = configs.n_patches_list[0]
        self.te_dim = configs.tpatchgnn_te_dim
        self.n_layer = configs.n_layers
        self.tf_layer = configs.factor
        self.K = configs.top_k

        # 1. Global Frequency Branch (NUDFT)
        self.global_nudft = GlobalRawNUDFT(d_model=self.hid_dim, num_freqs=self.num_freqs, n_vars=None)
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        self.seasonal_weight = nn.Parameter(torch.tensor(0.1))

        # 2. Time Encoding (TE) Generators
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, self.te_dim - 1)

        # 3. TTCN (Temporal Trend Convolution Network) — compute ttcn_dim first
        input_dim = 1 + self.te_dim
        ttcn_dim = self.hid_dim - 1
        self.ttcn_dim = ttcn_dim

        # 4. Node Embeddings — shape matches attention dim (ttcn_dim + 1)
        self.node_emb = nn.Parameter(torch.randn(1, 1, self.N, self.ttcn_dim + 1))
        self.Filter_Generators = nn.Sequential(
            nn.Linear(input_dim, ttcn_dim, bias=True), nn.ReLU(inplace=True),
            nn.Linear(ttcn_dim, ttcn_dim, bias=True), nn.ReLU(inplace=True),
            nn.Linear(ttcn_dim, input_dim * ttcn_dim, bias=True),
        )
        self.T_bias = nn.Parameter(torch.randn(1, ttcn_dim))

        # 5. Patch Attention
        self.query_patches = nn.Parameter(torch.randn(1, 1, self.K, self.ttcn_dim + 1))
        self.ADD_PE = PositionalEncoding(self.hid_dim)

        # 6. Mixer Layers (Dual-Mixing)
        self.patch_mixer_layers = nn.ModuleList()
        self.var_mixer_layers = nn.ModuleList()
        self.norm_patch = nn.ModuleList()
        self.norm_var = nn.ModuleList()
        for _ in range(self.n_layer):
            self.patch_mixer_layers.append(nn.Sequential(
                nn.Linear(self.K, self.K * self.tf_layer), nn.GELU(), nn.Dropout(self.dropout),
                nn.Linear(self.K * self.tf_layer, self.K), nn.Dropout(self.dropout),
            ))
            self.var_mixer_layers.append(nn.Sequential(
                nn.Linear(self.N, self.N * self.tf_layer), nn.GELU(), nn.Dropout(self.dropout),
                nn.Linear(self.N * self.tf_layer, self.N), nn.Dropout(self.dropout),
            ))
            self.norm_patch.append(nn.LayerNorm(self.hid_dim))
            self.norm_var.append(nn.LayerNorm(self.hid_dim))

        # 7. Output Layers
        self.temporal_agg = nn.Sequential(nn.Linear(self.hid_dim * self.K, self.hid_dim))
        self.decoder = nn.Sequential(
            nn.Linear(self.hid_dim + self.te_dim, self.hid_dim), nn.ReLU(inplace=True),
            nn.Linear(self.hid_dim, self.hid_dim), nn.ReLU(inplace=True),
            nn.Linear(self.hid_dim, 1),
        )

        # Normalization
        self.revin = MaskedRevIN(configs.enc_in) if configs.revin else None
        self.loss_fn = MSE(configs)

    def _LearnableTE(self, tt):
        return torch.cat([self.te_scale(tt), torch.sin(self.te_periodic(tt))], -1)

    def _TTCN(self, x_int, mask_x):
        n_dim, l_obs, _ = mask_x.shape
        filter_gen = self.Filter_Generators(x_int)
        filter_mask = filter_gen * mask_x + (1 - mask_x) * (-1e8)
        filter_seqnorm = torch.nn.functional.softmax(filter_mask, dim=-2)
        filter_seqnorm = filter_seqnorm.view(n_dim, l_obs, self.ttcn_dim, -1)
        x_int_broad = x_int.unsqueeze(dim=-2).repeat(1, 1, self.ttcn_dim, 1)
        ttcn_out = torch.sum(torch.sum(x_int_broad * filter_seqnorm, dim=-3), dim=-1)
        return torch.relu(ttcn_out + self.T_bias)

    def _process_local_branch(self, x, time_features, x_mask):
        b, l_obs, n_vars = x.shape
        remainder = l_obs % self.n_patch
        if remainder != 0:
            padding_len = self.n_patch - remainder
            pad_x = torch.zeros((b, padding_len, n_vars), device=x.device, dtype=x.dtype)
            pad_mask = torch.zeros((b, padding_len, n_vars), device=x.device, dtype=x_mask.dtype)
            pad_t = torch.zeros((b, padding_len, 1), device=x.device, dtype=time_features.dtype)
            x_padded = torch.cat([x, pad_x], dim=1)
            mask_padded = torch.cat([x_mask, pad_mask], dim=1)
            t_padded = torch.cat([time_features, pad_t], dim=1)
        else:
            x_padded, mask_padded, t_padded = x, x_mask, time_features
        l_padded = x_padded.shape[1]
        patch_len = l_padded // self.n_patch
        x_patches = x_padded.reshape(b, self.n_patch, patch_len, n_vars)
        mask_patches = mask_padded.reshape(b, self.n_patch, patch_len, n_vars)
        t_patches = t_padded.reshape(b, self.n_patch, patch_len, 1).repeat(1, 1, 1, n_vars)
        x_patches = x_patches.permute(0, 3, 1, 2).reshape(-1, patch_len, 1)
        mask_patches = mask_patches.permute(0, 3, 1, 2).reshape(-1, patch_len, 1)
        t_patches = t_patches.permute(0, 3, 1, 2).reshape(-1, patch_len, 1)
        te_his = self._LearnableTE(t_patches)
        x_with_te = torch.cat([x_patches, te_his], dim=-1)
        return self._imts_logic(x_with_te, mask_patches, b)

    def _imts_logic(self, x_with_te, mask_patches, b):
        # TTCN: convert raw patches to patch representations
        ttcn_out = self._TTCN(x_with_te, mask_patches)
        mask_patch_valid = mask_patches.sum(dim=1) > 0
        x_patch = torch.cat([ttcn_out, mask_patch_valid.float()], dim=-1)

        x_patch_flat = x_patch.view(b, self.N, self.n_patch, -1)
        node_identities = self.node_emb.permute(0, 2, 1, 3)
        x_patch_flat = x_patch_flat + node_identities
        x_patch_flat = x_patch_flat.view(b * self.N, self.n_patch, -1)
        x_patch_flat = self.ADD_PE(x_patch_flat).view(b, self.N, self.n_patch, -1)
        q = self.query_patches.expand(b, self.N, -1, -1)
        k_mat, v_mat = x_patch_flat, x_patch_flat
        scores = torch.matmul(q, k_mat.transpose(-1, -2)) / math.sqrt(x_patch_flat.shape[-1])
        attn_weights = torch.softmax(scores, dim=-1)
        x_patch_flat = torch.matmul(attn_weights, v_mat)
        x_curr = x_patch_flat
        for layer in range(self.n_layer):
            x_in = x_curr.permute(0, 1, 3, 2)
            x_out = self.patch_mixer_layers[layer](x_in).permute(0, 1, 3, 2)
            x_curr = self.norm_patch[layer](x_curr + x_out)
            x_in_var = x_curr.permute(0, 2, 3, 1)
            x_out_var = self.var_mixer_layers[layer](x_in_var).permute(0, 3, 1, 2)
            x_curr = self.norm_var[layer](x_curr + x_out_var)
        x_curr = x_curr.reshape(b, self.N, -1)
        x_curr = self.temporal_agg(x_curr)
        return x_curr

    def _decode_prediction(self, h, y_mark):
        b, n_vars, _ = h.shape
        time_steps_to_predict = y_mark[:, :, [0]]
        l_pred = time_steps_to_predict.shape[1]
        h_expanded = h.unsqueeze(dim=-2).repeat(1, 1, l_pred, 1)
        time_steps_exp = time_steps_to_predict.view(b, 1, l_pred, 1).repeat(1, n_vars, 1, 1)
        te_pred = self._LearnableTE(time_steps_exp)
        decoder_input = torch.cat([h_expanded, te_pred], dim=-1)
        outputs_raw = self.decoder(decoder_input)
        return outputs_raw.squeeze(-1).permute(0, 2, 1)

    def _add_seasonal_extrapolation(self, base_prediction, raw_spectrum, y_mark):
        b, l_pred, n_vars = base_prediction.shape
        time_steps_to_predict = y_mark[:, :, [0]]
        t_pred_flat = time_steps_to_predict.repeat(1, 1, n_vars).permute(0, 2, 1).reshape(b * n_vars, -1, 1)
        seasonal_forecast_flat = self.global_nudft.forecast(raw_spectrum, t_pred_flat)
        seasonal_forecast = seasonal_forecast_flat.view(b, n_vars, -1).permute(0, 2, 1)
        return base_prediction + self.seasonal_weight * seasonal_forecast

    def _compute_reconstruction_loss(self, raw_spectrum, x_flat, t_flat, mask_flat):
        recon_history_flat = self.global_nudft.forecast(raw_spectrum, t_flat)
        recon_err = torch.abs(recon_history_flat - x_flat)
        recon_err_masked = recon_err * mask_flat
        valid_points = mask_flat.sum()
        return recon_err_masked.sum() / (valid_points + 1e-5)

    def forward(
        self,
        x: Tensor,
        x_mark: Tensor | None = None,
        x_mask: Tensor | None = None,
        y: Tensor | None = None,
        y_mark: Tensor | None = None,
        y_mask: Tensor | None = None,
        **kwargs,
    ) -> dict:
        # BEGIN adaptor
        BATCH_SIZE, SEQ_LEN, ENC_IN = x.shape
        Y_LEN = self.configs.pred_len if self.configs.pred_len != 0 else SEQ_LEN
        if x_mark is None:
            x_mark = repeat(
                torch.arange(end=x.shape[1], dtype=x.dtype, device=x.device)
                / x.shape[1],
                "L -> B L 1",
                B=x.shape[0],
            )
        if x_mask is None:
            x_mask = torch.ones_like(x, device=x.device, dtype=x.dtype)
        if y is None:
            logger.warning(
                f"y is missing for the model input. This is only reasonable when the model is testing flops!"
            )
            y = torch.ones((BATCH_SIZE, Y_LEN, ENC_IN), dtype=x.dtype, device=x.device)
        if y_mark is None:
            y_mark = repeat(
                torch.arange(end=y.shape[1], dtype=y.dtype, device=y.device)
                / y.shape[1],
                "L -> B L 1",
                B=y.shape[0],
            )
        if y_mask is None:
            y_mask = torch.ones_like(y, device=y.device, dtype=y.dtype)
        # END adaptor

        b, l_obs, n_vars = x.shape

        # Apply RevIN normalization if enabled
        if self.revin is not None:
            x = self.revin(x, x_mask, mode="norm")

        # 1. Prepare global inputs (flattened for NUDFT)
        time_features = x_mark[:, :, [0]].repeat(1, 1, n_vars)
        x_flat = x.permute(0, 2, 1).reshape(b * n_vars, l_obs, 1)
        t_flat = time_features.permute(0, 2, 1).reshape(b * n_vars, l_obs, 1)
        mask_flat = x_mask.permute(0, 2, 1).reshape(b * n_vars, l_obs, 1)

        # 2. Global Frequency Branch (NUDFT)
        h_freq_flat, raw_spectrum = self.global_nudft(x_flat, t_flat, mask_flat)
        h_freq = h_freq_flat.view(b, n_vars, -1)

        # 3. Local Time Branch (Patch Processing)
        time_features_single = x_mark[:, :, [0]]
        h_time = self._process_local_branch(x, time_features_single, x_mask)

        # 4. Feature Fusion
        h = h_time + self.fusion_weight * h_freq

        if self.configs.task_name in ["long_term_forecast", "short_term_forecast"]:
            # 5. Decoding (Base Prediction)
            outputs = self._decode_prediction(h, y_mark)

            # 6. Seasonal Extrapolation
            outputs = self._add_seasonal_extrapolation(outputs, raw_spectrum, y_mark)

            f_dim = -1 if self.configs.features == "MS" else 0
            output = {
                "pred": outputs[:, :, f_dim:],
                "true": y[:, :, f_dim:],
                "mask": y_mask[:, :, f_dim:]
            }

            # 7. Compute loss
            loss = self.loss_fn(**output)["loss"] + self.configs.mtan_alpha * self._compute_reconstruction_loss(raw_spectrum, x_flat, t_flat, mask_flat)
            output["loss"] = loss
            return output
        else:
            raise NotImplementedError