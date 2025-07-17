import torch
import torch.nn as nn

from utils.ExpConfigs import ExpConfigs

class Loss(nn.Module):
    def __init__(self, configs:ExpConfigs):
        super(Loss, self).__init__()

    def forward(self, pred, true, mask=None, **kwargs):
        if mask is None:
            mask = torch.ones_like(true, device=true.device)

        residual = (pred - true) * mask
        num_eval = mask.sum()

        return {
            "loss": (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        }