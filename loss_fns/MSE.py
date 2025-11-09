# Code from: https://github.com/Ladbaby/PyOmniTS
import torch
import torch.nn as nn
from torch import Tensor

from utils.ExpConfigs import ExpConfigs

class Loss(nn.Module):
    def __init__(self, configs:ExpConfigs):
        super(Loss, self).__init__()

    def forward(
        self, 
        pred: Tensor, 
        true: Tensor, 
        mask: Tensor | None = None, 
        **kwargs
    ) -> dict[str, Tensor]:
        # BEGIN adaptor
        if mask is None:
            mask = torch.ones_like(true, device=true.device)
        # END adaptor

        residual = (pred - true) * mask
        squared_error = residual ** 2
        num_eval = mask.sum()
        loss = squared_error.sum() / (num_eval if num_eval > 0 else 1)

        return {
            "loss": loss,
            "loss_reduction_none": squared_error # same shape as pred
        }