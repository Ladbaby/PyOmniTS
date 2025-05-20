import numpy as np
import torch

__all__ = ['get_mask']

def get_mask(mask):
    if mask == 'none':
        return none()
    elif mask == 'ordered_right_half' or mask == 'ordered_0':
        return ordered(ratio_zero=0.5, right_zero=False)
    elif mask == 'ordered_left_half' or mask == 'ordered_1':
        return ordered(ratio_zero=0.5, right_zero=True)
    elif mask == 'random_half':
        return random(ratio_zero=0.5)
    elif mask == 'parity_even':
        return parity(even_zero=False)
    elif mask == 'parity_odd':
        return parity(even_zero=True)
    else:
        raise NotImplementedError()

def none():
    return lambda _: torch.Tensor([0])

def random(ratio_zero=0.5):
    def mask(dim):
        if dim == 1:
            return torch.Tensor([1])
        mask = np.zeros(dim)
        size = np.clip(int(dim * ratio_zero), 1, dim - 1)
        mask[np.random.choice(np.arange(dim), size, replace=False)] = 1
        return torch.Tensor(mask)
    return mask

def ordered(ratio_zero=0.5, right_zero=False):
    def mask(dim):
        if dim == 1:
            return torch.Tensor([1])
        mask = np.ones(dim)
        size = np.clip(int(dim * ratio_zero), 1, dim - 1)
        mask[:size] = 0
        if right_zero:
            mask = 1 - mask
        return torch.Tensor(mask)
    return mask

def parity(even_zero=False):
    # If even zero, first element (ind=0) will be 0, and so on
    def mask(dim):
        if dim == 1:
            return torch.Tensor([1])
        mask = np.ones(dim)
        mask[::2] = 0
        if even_zero:
            mask = 1 - mask
        return torch.Tensor(mask)
    return mask