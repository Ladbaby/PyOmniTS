from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch import Tensor

from utils.ExpConfigs import ExpConfigs

logger = logging.getLogger(__name__)


class _GRUwECell(nn.Module):
    """ GRUwE cell definition
    """

    def __init__(self, n_features: int, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size

        token_input_dim = 3 * hidden_size
        gate_input_dim = token_input_dim + hidden_size

        self.W_z = nn.Linear(gate_input_dim, hidden_size)
        self.W_r = nn.Linear(gate_input_dim, hidden_size)
        self.W_h = nn.Linear(gate_input_dim, hidden_size)
        self.W_decay = nn.Linear(1, hidden_size)
        self.hidden_to_obs = nn.Linear(hidden_size, n_features)

        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in [self.W_z, self.W_r, self.W_h, self.W_decay, self.hidden_to_obs]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def init_hidden(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

    def decay_hidden(self, h: Tensor, delta_t: Tensor) -> Tensor:
        if delta_t.dim() == 1:
            gamma = torch.exp(-torch.relu(self.W_decay(delta_t.unsqueeze(-1))))
            return h * gamma

        if delta_t.dim() == 2:
            b, t = delta_t.shape
            gamma = torch.exp(-torch.relu(self.W_decay(delta_t.reshape(b * t, 1)))).reshape(b, t, self.hidden_size)
            return h.unsqueeze(1) * gamma

        raise ValueError(f"delta_t must be rank-1 or rank-2, got shape {tuple(delta_t.shape)}")

    def project(self, h: Tensor) -> Tensor:
        if h.dim() in (2, 3):
            return self.hidden_to_obs(h)
        raise ValueError(f"h must be rank-2 or rank-3, got shape {tuple(h.shape)}")

    def update(
        self,
        h: Tensor,
        source_h: Tensor,
        value_emb: Tensor,
        mask_emb: Tensor,
        update_mask: Tensor | None = None,
    ) -> Tensor:
        token = torch.cat((source_h, value_emb, mask_emb), dim=-1)
        combined = torch.cat((token, h), dim=-1)

        z = torch.sigmoid(self.W_z(combined))
        r = torch.sigmoid(self.W_r(combined))
        combined_r = torch.cat((token, r * h), dim=-1)
        h_tilde = torch.tanh(self.W_h(combined_r))
        h_new = (1.0 - z) * h + z * h_tilde
        h_new = self.dropout(h_new)

        if update_mask is not None:
            mask = update_mask.bool()
            if mask.dim() == 1:
                mask = mask.unsqueeze(-1)
            h_new = torch.where(mask, h_new, h)

        return h_new


class Model(nn.Module):
    """ GRUwE Model

    Paper: Still Competitive: Revisiting Recurrent Models for Irregular Time Series Prediction
    Venue: TMLR 2026

    Supported tasks:
    - short_term_forecast
    - long_term_forecast
    """
    def __init__(self, configs: ExpConfigs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        if self.task_name not in ["short_term_forecast", "long_term_forecast"]:
            raise NotImplementedError(f"{self.task_name} not implemented for GRUwE")

        self.enc_in = configs.enc_in
        self.hidden_size = int(configs.d_model)
        self.num_cells = int(configs.n_layers)
        if self.num_cells < 1:
            raise ValueError(f"n_layers must be >= 1, got {self.num_cells}")

        self.value_proj = nn.Linear(self.enc_in, self.hidden_size)
        self.mask_proj = nn.Linear(self.enc_in, self.hidden_size)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.mask_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.zeros_(self.mask_proj.bias)

        self.cells = nn.ModuleList(
            [
                _GRUwECell(
                    n_features=self.enc_in,
                    hidden_size=self.hidden_size,
                    dropout=float(configs.dropout),
                )
                for _ in range(self.num_cells)
            ]
        )

    def _default_x_mark(self, x: Tensor) -> Tensor:
        b, l, _ = x.shape
        return torch.linspace(0.0, 1.0, steps=l, device=x.device, dtype=x.dtype).view(1, l, 1).repeat(b, 1, 1)

    def _prepare_inputs(
        self,
        x: Tensor,
        x_mark: Tensor | None,
        x_mask: Tensor | None,
        y: Tensor | None,
        y_mark: Tensor | None,
        y_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch_size, _, enc_in = x.shape
        if enc_in != self.enc_in:
            raise ValueError(f"Expected x.shape[-1] == enc_in == {self.enc_in}, got {enc_in}")

        if x_mark is None:
            x_mark = self._default_x_mark(x)
        if x_mask is None:
            x_mask = torch.ones_like(x)

        if y is None:
            pred_len = self.configs.pred_len_max_irr or self.configs.pred_len
            y = torch.zeros(batch_size, pred_len, enc_in, device=x.device, dtype=x.dtype)
            if self.task_name != "imputation":
                logger.warning("y is missing for the model input. This is only reasonable when testing FLOPs.")
        if y_mark is None:
            if self.task_name == "imputation":
                y_mark = x_mark.clone()
            else:
                y_mark = self._default_x_mark(y)
        if y_mask is None:
            y_mask = torch.ones_like(y)

        x_mark = x_mark[..., :1].to(dtype=x.dtype)
        y_mark = y_mark[..., :1].to(dtype=x.dtype)
        x_mask = x_mask.to(dtype=x.dtype)
        y_mask = y_mask.to(dtype=x.dtype)
        return x, x_mark, x_mask, y, y_mark, y_mask

    def _embed_value(self, x_t: Tensor) -> Tensor:
        return self.value_proj(x_t)

    def _embed_mask(self, mask_t: Tensor) -> Tensor:
        return self.mask_proj(mask_t)

    def _init_hidden_list(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> list[Tensor]:
        return [cell.init_hidden(batch_size, device, dtype) for cell in self.cells]

    def _encode_history(self, x: Tensor, x_mark: Tensor, x_mask: Tensor) -> tuple[list[Tensor], Tensor, Tensor]:
        batch_size, seq_len, _ = x.shape
        h_list = self._init_hidden_list(batch_size, x.device, x.dtype)
        hist_pred = torch.zeros_like(x)
        prev_time = torch.zeros(batch_size, device=x.device, dtype=x.dtype)

        for t in range(seq_len):
            current_time = x_mark[:, t, 0]
            delta_t = torch.clamp(current_time - prev_time, min=0.0)

            layer_mask = x_mask[:, t, :]
            update_mask = layer_mask.any(dim=-1, keepdim=True)

            for layer_idx, cell in enumerate(self.cells):
                h_decayed = cell.decay_hidden(h_list[layer_idx], delta_t)
                h_list[layer_idx] = torch.where(
                    update_mask,
                    h_decayed,
                    h_list[layer_idx],
                )

            hist_pred[:, t, :] = self.cells[-1].project(h_list[-1])

            raw_x_t = x[:, t, :] * x_mask[:, t, :]

            value_emb_t = self._embed_value(raw_x_t)
            mask_emb_t = self._embed_mask(layer_mask)

            source_h = torch.zeros_like(h_list[0])
            h_list[0] = self.cells[0].update(
                h=h_list[0],
                source_h=source_h,
                value_emb=value_emb_t,
                mask_emb=mask_emb_t,
                update_mask=update_mask,
            )

            for layer_idx in range(1, self.num_cells):
                h_list[layer_idx] = self.cells[layer_idx].update(
                    h=h_list[layer_idx],
                    source_h=h_list[layer_idx - 1],
                    value_emb=value_emb_t,
                    mask_emb=mask_emb_t,
                    update_mask=update_mask,
                )

            prev_time = torch.where(
                update_mask.squeeze(-1),
                current_time,
                prev_time,
            )

        return h_list, hist_pred, prev_time


    def forecast(
        self,
        x: Tensor,
        x_mark: Tensor,
        x_mask: Tensor,
        y_mark: Tensor,
    ) -> Tensor:
        # Keep the timestamp of the final historical observation so that the
        # first forecast decay uses the gap from history to the first target.
        h_list, _, prev_time = self._encode_history(
            x,
            x_mark,
            x_mask,
        )

        future_times = y_mark[:, :, 0]
        batch_size, pred_len = future_times.shape

        zero_x = torch.zeros(
            batch_size,
            self.enc_in,
            device=x.device,
            dtype=x.dtype,
        )
        zero_mask = torch.zeros_like(zero_x)

        zero_value_emb = self._embed_value(zero_x)
        zero_mask_emb = self._embed_mask(zero_mask)

        dec_out = torch.zeros(
            batch_size,
            pred_len,
            self.enc_in,
            device=x.device,
            dtype=x.dtype,
        )

        for t in range(pred_len):
            current_time = future_times[:, t]

            # Elapsed time from the previous historical/forecast timestamp.
            delta_t = torch.clamp(
                current_time - prev_time,
                min=0.0,
            )

            # Apply the same future-time decay used by GRUwE_IND.
            for layer_idx, cell in enumerate(self.cells):
                h_list[layer_idx] = cell.decay_hidden(
                    h_list[layer_idx],
                    delta_t,
                )

            # advance the state with a missing-input transition
            # this mimics the original implementation
            source_h = torch.zeros_like(h_list[0])
            h_list[0] = self.cells[0].update(
                h=h_list[0],
                source_h=source_h,
                value_emb=zero_value_emb,
                mask_emb=zero_mask_emb,
                update_mask=None,
            )

            for layer_idx in range(1, self.num_cells):
                h_list[layer_idx] = self.cells[layer_idx].update(
                    h=h_list[layer_idx],
                    source_h=h_list[layer_idx - 1],
                    value_emb=zero_value_emb,
                    mask_emb=zero_mask_emb,
                    update_mask=None,
                )

            dec_out[:, t, :] = self.cells[-1].project(
                h_list[-1]
            )

            prev_time = current_time

        return dec_out

    def forward(
        self,
        x: Tensor,
        x_mark: Tensor | None = None,
        x_mask: Tensor | None = None,
        y: Tensor | None = None,
        y_mark: Tensor | None = None,
        y_mask: Tensor | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        x, x_mark, x_mask, y, y_mark, y_mask = self._prepare_inputs(x, x_mark, x_mask, y, y_mark, y_mask)

        if self.task_name in ["short_term_forecast", "long_term_forecast"]:
            dec_out = self.forecast(x, x_mark, x_mask, y_mark)
        else:
            raise NotImplementedError(f"{self.task_name} not implemented for GRUwE")

        f_dim = -1 if self.configs.features == "MS" else 0
        return {
            "pred": dec_out[:, :, f_dim:],
            "true": y[:, :, f_dim:],
            "mask": y_mask[:, :, f_dim:],
        }

