import math
import torch
import torch.nn as nn


class MaskedRevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False):
        super(MaskedRevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x, mask):
        valid_count = mask.sum(dim=1, keepdim=True)
        valid_count = torch.clamp(valid_count, min=1.0)
        self.mean = (x * mask).sum(dim=1, keepdim=True) / valid_count
        var = ((x - self.mean) ** 2 * mask).sum(dim=1, keepdim=True) / valid_count
        self.stdev = torch.sqrt(var + self.eps)
        self.mean = self.mean.detach()
        self.stdev = self.stdev.detach()

    def forward(self, x, mask, mode: str):
        if mode == "norm":
            self._get_statistics(x, mask)
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
        elif mode == "denorm":
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + 1e-10)
            x = x * self.stdev + self.mean
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class GlobalRawNUDFT(nn.Module):
    def __init__(self, d_model: int, num_freqs: int = 64, n_vars: int = None):
        super(GlobalRawNUDFT, self).__init__()
        self.num_freqs = num_freqs
        self.d_model = d_model
        self.n_vars = n_vars
        self.freqs = nn.Parameter(
            torch.arange(1, num_freqs + 1).float(), requires_grad=True
        )
        if n_vars is not None:
            self.var_mixer = nn.Sequential(
                nn.Linear(n_vars, n_vars),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(n_vars, n_vars),
            )
        else:
            self.var_mixer = None
        self.spectrum_encoder = nn.Sequential(
            nn.Linear(num_freqs * 2, num_freqs * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(num_freqs * 2, num_freqs * 2),
        )
        self.freq_mlp = nn.Sequential(
            nn.Linear(num_freqs * 2, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, t: torch.Tensor, mask: torch.Tensor):
        args = 2.0 * math.pi * t * self.freqs.view(1, 1, -1)
        cos_basis = torch.cos(args)
        sin_basis = torch.sin(args)
        x_masked = x * mask
        norm_factor = torch.clamp(mask.sum(dim=1), min=5.0)
        real_spectrum = torch.sum(x_masked * cos_basis, dim=1) / norm_factor
        imag_spectrum = torch.sum(x_masked * -sin_basis, dim=1) / norm_factor
        spectrum = torch.cat([real_spectrum, imag_spectrum], dim=-1)
        spectrum = self.spectrum_encoder(spectrum)
        if self.var_mixer is not None:
            BN, D_freq = spectrum.shape
            N = self.n_vars
            B = BN // N
            spectrum_reshaped = spectrum.view(B, N, D_freq)
            spectrum_transposed = spectrum_reshaped.permute(0, 2, 1)
            mixed_spectrum = self.var_mixer(spectrum_transposed)
            spectrum = (
                (spectrum_transposed + mixed_spectrum)
                .permute(0, 2, 1)
                .reshape(BN, D_freq)
            )
        raw_spectrum = spectrum
        return self.norm(self.freq_mlp(spectrum)), raw_spectrum

    def forecast(self, spectrum: torch.Tensor, t_future: torch.Tensor) -> torch.Tensor:
        real_spectrum = spectrum[:, : self.num_freqs].unsqueeze(1)
        imag_spectrum = spectrum[:, self.num_freqs :].unsqueeze(1)
        args = 2.0 * math.pi * t_future * self.freqs.view(1, 1, -1)
        cos_basis = torch.cos(args)
        sin_basis = torch.sin(args)
        pred = torch.sum(real_spectrum * cos_basis, dim=-1) + torch.sum(
            imag_spectrum * -sin_basis, dim=-1
        )
        return pred.unsqueeze(-1)
