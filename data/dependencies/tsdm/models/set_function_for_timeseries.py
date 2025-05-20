r"""#TODO add module summary line.

#TODO add module description.
"""

__all__ = [
    # CLASSES
    "SetFuncTS",
    "GroupedSetFuncTS",
]

from typing import Optional

import torch
from torch import Tensor, jit, nn

from data.dependencies.tsdm.encoders.torch import PositionalEncoder, Time2Vec
from data.dependencies.tsdm.models.generic import (
    MLP,
    DeepSet,
    DeepSetReZero,
    ReZeroMLP,
    ScaledDotProductAttention,
)
from data.dependencies.tsdm.utils.decorators import autojit


@autojit
class SetFuncTS(nn.Module):
    r"""Set function for time series.

    Attributes
    ----------
    time_encoder: nn.Module, default PositionalEncoder
        Signature: ``(..., *N) -> (..., *N, dₜ)``
    key_encoder: nn.Module, default DeepSet
        Signature ``(..., *N, K) -> (..., *N, dₖ)``
    value_encoder: nn.Module, default MLP
        Signature: ``(..., *N, V) -> (..., *N, dᵥ)``
    attention: nn.Module, default ScaledDotProductAttention
        Signature: ``(..., *N, dₖ), (..., *N, dᵥ) -> (..., F)``
    head: nn.Module, default MLP
        Signature: ``(..., F) -> (..., E)``

    References
    ----------
    - | Set Functions for Time Series
      | Max Horn, Michael Moor, Christian Bock, Bastian Rieck, Karsten Borgwardt
      | Proceedings of the 37th International Conference on Machine Learning
      | PMLR 119:4353-4363, 2020.
      | https://proceedings.mlr.press/v119/horn20a.html
      | https://github.com/BorgwardtLab/Set_Functions_for_Time_Series
    """

    HP: dict = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__doc__": __doc__,
        "__module__": __module__,  # type: ignore[name-defined]
        "time_encoder": PositionalEncoder.HP,
        "key_encoder": DeepSet.HP,
        "value_encoder": MLP.HP,
        "attention": ScaledDotProductAttention.HP,
        "head": MLP.HP,
    }
    r"""Dictionary of hyperparameters."""

    # BUFFER
    dummy: Tensor

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        latent_size: Optional[int] = None,
        dim_keys: Optional[int] = None,
        dim_vals: Optional[int] = None,
        dim_time: Optional[int] = None,
        dim_deepset: Optional[int] = None,
    ) -> None:
        r"""Initialize the model.

        Parameters
        ----------
        input_size: int,
        output_size: int,
        latent_size: Optional[int] = None,
        dim_keys: Optional[int] = None,
        dim_vals: Optional[int] = None,
        dim_time: Optional[int] = None,
        dim_deepset: Optional[int] = None,
        """
        super().__init__()

        dim_keys = input_size if dim_keys is None else dim_keys
        dim_vals = input_size if dim_vals is None else dim_vals
        dim_time = 8 if dim_time is None else dim_time
        latent_size = input_size if latent_size is None else latent_size
        # time_encoder
        # feature_encoder -> CNN?
        self.time_encoder = PositionalEncoder(dim_time, scale=10.0)
        self.key_encoder = DeepSet(
            input_size + dim_time - 1,
            dim_keys,
            latent_size=dim_deepset,
            hidden_size=dim_deepset,
        )
        self.value_encoder = MLP(
            input_size + dim_time - 1, dim_vals, hidden_size=dim_vals
        )
        self.attention = ScaledDotProductAttention(
            dim_keys + input_size + dim_time - 1, dim_vals, latent_size
        )
        self.head = MLP(latent_size, output_size)
        self.register_buffer("dummy", torch.zeros(1))

    @jit.export
    def forward(self, t: Tensor, v: Tensor, m: Tensor) -> Tensor:
        r""".. Signature: ``[(*N, dₜ), (*N, dᵥ), (*N, dₘ)] -> (..., F)``.

        s must be a tensor of the shape $L×(2+C)4, $sᵢ = [tᵢ, zᵢ, mᵢ]$, where

        - $tᵢ$ is timestamp
        - $zᵢ$ is observed value
        - $mᵢ$ is identifier

        C is the number of classes (one-hot encoded identifier)

        Parameters
        ----------
        t: Tensor
        v: Tensor
        m: Tensor

        Returns
        -------
        Tensor
        """
        t = t.to(device=self.dummy.device)
        v = v.to(device=self.dummy.device)
        m = m.to(device=self.dummy.device)

        time_features = self.time_encoder(t)

        if v.ndim < m.ndim:
            v = v.unsqueeze(-1)

        s = torch.cat([time_features, v, m], dim=-1)
        fs = self.key_encoder(s)
        fs = torch.tile(fs.unsqueeze(-2), (s.shape[-2], 1))
        K = torch.cat([fs, s], dim=-1)
        V = self.value_encoder(s)
        mask = torch.isnan(s[..., 0])
        z = self.attention(K, V, mask=mask)
        y = self.head(z)
        return y

    @jit.export
    def forward_tuple(self, t: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        r""".. Signature: ``[(*N, dₜ), (*N, dᵥ), (*N, dₘ)] -> F``."""
        return self.forward(t[0], t[1], t[2])

    @jit.export
    def forward_batch(self, batch: list[tuple[Tensor, Tensor, Tensor]]) -> Tensor:
        r""".. Signature: ``[...,  [(*N, dₜ), (*N, dᵥ), (*N, dₘ)] -> (..., F)``.

        Parameters
        ----------
        batch: list[tuple[Tensor, Tensor, Tensor]]

        Returns
        -------
        Tensor
        """
        return torch.cat([self.forward(t, v, m) for t, v, m in batch])


@autojit
class GroupedSetFuncTS(nn.Module):
    r"""Set function for time series.

    Attributes
    ----------
    time_encoder: nn.Module, default PositionalEncoder
        Signature: ``(..., *N) -> (..., *N, dₜ)``
    key_encoder: nn.Module, default DeepSet
        Signature: ``(..., *N, K) -> (..., *N, dₖ)``
    value_encoder: nn.Module, default MLP
        Signature: ``(..., *N, V) -> (..., *N, dᵥ)``
    attention: nn.Module, default ScaledDotProductAttention
        Signature: ``(..., *N, dₖ), (..., *N, dᵥ) -> (..., F)``
    head: nn.Module, default MLP
        Signature: ``(..., F) -> (..., E)``

    References
    ----------
    - | Set Functions for Time Series
      | Max Horn, Michael Moor, Christian Bock, Bastian Rieck, Karsten Borgwardt
      | Proceedings of the 37th International Conference on Machine Learning
      | PMLR 119:4353-4363, 2020.
      | https://proceedings.mlr.press/v119/horn20a.html
      | https://github.com/BorgwardtLab/Set_Functions_for_Time_Series
    """

    HP: dict = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__doc__": __doc__,
        "__module__": __module__,  # type: ignore[name-defined]
        "time_encoder": PositionalEncoder.HP,
        "key_encoder": DeepSet.HP,
        "value_encoder": MLP.HP,
        "attention": ScaledDotProductAttention.HP,
        "head": MLP.HP,
    }
    r"""Dictionary of hyperparameters."""

    # BUFFER
    ZERO: Tensor

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        fast_encoder: nn.Module,
        slow_encoder: nn.Module,
        latent_size: Optional[int] = None,
        dim_keys: Optional[int] = None,
        dim_vals: Optional[int] = None,
        dim_time: Optional[int] = None,
        dim_deepset: Optional[int] = None,
    ) -> None:
        r"""Initialize the model.

        Parameters
        ----------
        input_size: int,
        output_size: int,
        latent_size: Optional[int] = None,
        dim_keys: Optional[int] = None,
        dim_vals: Optional[int] = None,
        dim_time: Optional[int] = None,
        dim_deepset: Optional[int] = None,
        """
        super().__init__()

        dim_keys = input_size if dim_keys is None else dim_keys
        dim_vals = input_size if dim_vals is None else dim_vals
        dim_time = 8 if dim_time is None else dim_time
        latent_size = input_size if latent_size is None else latent_size
        # time_encoder
        # feature_encoder -> CNN?
        self.fast_encoder = fast_encoder
        self.slow_encoder = slow_encoder

        self.time_encoder = Time2Vec(dim_time)
        self.key_encoder = DeepSetReZero(
            input_size,
            dim_keys,
            latent_size=dim_deepset,
            hidden_size=dim_deepset,
        )

        self.value_encoder = ReZeroMLP(input_size, dim_vals, latent_size=dim_vals)
        # self.value_encoder = MLP(input_size, dim_vals, hidden_size=dim_vals)

        self.attention = ScaledDotProductAttention(
            dim_keys + input_size, dim_vals, latent_size
        )
        self.head = ReZeroMLP(latent_size, output_size)
        # self.head = MLP(latent_size, output_size)

        self.register_buffer("ZERO", torch.tensor(0.0))

    @jit.export
    def forward(self, slow: Tensor, fast: Tensor) -> Tensor:
        r""".. Signature:: ``[(*N, dₜ), (*N, dᵥ), (*N, dₘ)] -> (..., F)``.

        s must be a tensor of the shape $L×(2+C)$, $sᵢ = [tᵢ, zᵢ, mᵢ]$, where
        - $tᵢ$ is timestamp
        - $zᵢ$ is observed value
        - $mᵢ$ is identifier

        C is the number of classes (one-hot encoded identifier)

        Parameters
        ----------
        fast: Tensor
        slow: Tensor

        Returns
        -------
        Tensor
        """
        fast = fast.to(device=self.ZERO.device)
        slow = slow.to(device=self.ZERO.device)

        t_slow = slow[..., 0]
        t_fast = fast[..., 0]

        fast = fast[..., 1:]
        slow = slow[..., 1:]

        time_features_slow = self.time_encoder(t_slow)  # [..., ] -> [..., dₜ]
        time_features_fast = self.time_encoder(t_fast)  # [..., ] -> [..., dₜ]

        slow = torch.cat(
            [time_features_slow, slow], dim=-1
        )  # [..., d] -> [..., d+dₜ-1]
        fast = torch.cat(
            [time_features_fast, fast], dim=-1
        )  # [..., d] -> [..., d+dₜ-1]

        # FIXME: https://github.com/pytorch/pytorch/issues/73291
        torch.cuda.synchronize()  # needed when cat holds 0-size tensor

        slow = self.slow_encoder(slow)

        fast = fast.swapaxes(-1, -2)
        if fast.ndim == 2:
            fast = self.fast_encoder(fast.unsqueeze(0)).squeeze(0)
        else:
            fast = self.fast_encoder(fast)
        fast = fast.swapaxes(-1, -2)

        s = torch.cat([slow, fast], dim=-2)

        fs = self.key_encoder(s)
        fs = torch.tile(fs.unsqueeze(-2), (s.shape[-2], 1))
        K = torch.cat([fs, s], dim=-1)
        V = self.value_encoder(s)
        mask = torch.isnan(s[..., 0])

        z = self.attention(K, V, mask=mask)
        y = self.head(z)
        return y.squeeze()

    @jit.export
    def forward_batch(self, batch: list[tuple[Tensor, Tensor]]) -> Tensor:
        r""".. Signature:: ``[...,  [(*N, dₜ), (*N, dᵥ), (*N, dₘ)]] -> (..., F)``.

        Parameters
        ----------
        batch: list[tuple[Tensor, Tensor, Tensor]]

        Returns
        -------
        Tensor
        """
        return torch.stack([self.forward(slow, fast) for slow, fast in batch])

    @jit.export
    def forward_padded(self, batch: list[tuple[Tensor, Tensor]]) -> Tensor:
        r""".. Signature:: ``[...,  [(*N, dₜ), (*N, dᵥ), (*N, dₘ)]] -> (..., F)``.

        Parameters
        ----------
        batch: list[tuple[Tensor, Tensor, Tensor]]

        Returns
        -------
        Tensor
        """
        # X, Y = list(zip(*batch))
        X = []
        Y = []
        for x, y in batch:
            X.append(x)
            Y.append(y)

        x = torch.nn.utils.rnn.pad_sequence(
            X, batch_first=True, padding_value=float("nan")
        )
        y = torch.nn.utils.rnn.pad_sequence(
            Y, batch_first=True, padding_value=float("nan")
        )
        return self.forward(x, y)
