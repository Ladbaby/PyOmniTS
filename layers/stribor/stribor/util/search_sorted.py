import torch


def searchsorted(bin_locations, inputs, eps=1e-06):
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1
