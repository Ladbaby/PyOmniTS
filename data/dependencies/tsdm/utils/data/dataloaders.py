r"""General Purpose Data Loaders for Time Series Data.

We implement multiple levels of abstraction.

- Dataloader for TimeSeriesTensor
- Dataloader for tuple of TimeSeriesTensor
- Dataloader for MetaDataset
   - sample dataset by index, then sample from that dataset.
"""

__all__ = [
    # Functions
    "collate_list",
    "collate_packed",
    "collate_padded",
    "unpad_sequence",
    "unpack_sequence",
]

from typing import Optional

import torch
from torch import Tensor
from torch.nn.utils.rnn import (
    PackedSequence,
    pack_sequence,
    pad_packed_sequence,
    pad_sequence,
)

from data.dependencies.tsdm.linalg import aggregate_and, cumulative_and


def collate_list(batch: list[Tensor]) -> list[Tensor]:
    r"""Collates list of tensors as list of tensors."""
    return batch


def collate_packed(batch: list[Tensor]) -> PackedSequence:
    r"""Collates list of tensors into a PackedSequence."""
    # First, need to sort in descending order by length
    batch.sort(key=Tensor.__len__, reverse=True)
    return pack_sequence(batch)


def collate_padded(
    batch: list[Tensor], batch_first: bool = True, padding_value: float = 0.0
) -> Tensor:
    r"""Collates list of tensors of varying lengths into a single Tensor, padded with zeros.

    Equivalent to `torch.nn.utils.rnn.pad_sequence`, but with `batch_first=True` as default

    .. Signature:: ``[ (lᵢ, ...)_{i=1:B} ] -> (B, lₘₐₓ, ...)``.

    Parameters
    ----------
    batch: list[Tensor]
    batch_first: bool, default True
    padding_value: float, default True

    Returns
    -------
    Tensor
    """
    return pad_sequence(batch, batch_first=batch_first, padding_value=padding_value)


def unpack_sequence(batch: PackedSequence) -> list[Tensor]:
    r"""Reverse operation of pack_sequence."""
    batch_pad_packed, lengths = pad_packed_sequence(batch, batch_first=True)
    return [x[:l] for x, l in zip(batch_pad_packed, lengths)]


def unpad_sequence(
    padded_seq: Tensor,
    batch_first: bool = False,
    lengths: Optional[Tensor] = None,
    padding_value: float = 0.0,
) -> list[Tensor]:
    r"""Reverse operation of `torch.nn.utils.rnn.pad_sequence`."""
    padded_seq = padded_seq.swapaxes(0, 1) if not batch_first else padded_seq
    padding: Tensor = torch.tensor(
        padding_value, dtype=padded_seq.dtype, device=padded_seq.device
    )

    if lengths is not None:
        return [x[0:l] for x, l in zip(padded_seq, lengths)]

    # infer lengths from mask
    if torch.isnan(padding):
        mask = torch.isnan(padded_seq)
    else:
        mask = padded_seq == padding_value

    # all features are masked
    dims: list[int] = list(range(min(2, padded_seq.ndim), padded_seq.ndim))
    agg = aggregate_and(mask, dim=dims)
    # count, starting from the back, until the first observation occurs.
    inferred_lengths = (~cumulative_and(agg.flip(dims=(1,)), dim=1)).sum(dim=1)

    return [x[0:l] for x, l in zip(padded_seq, inferred_lengths)]
