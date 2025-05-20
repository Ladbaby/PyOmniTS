from itertools import chain
import torch
__all__ = ['flatten_params']


def _flatten(sequence):
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


def flatten_params(*nets):
    return _flatten(chain(*[x.parameters() for x in nets]))
