# Code from: https://github.com/Ladbaby/PyOmniTS
import torch
import torch.nn as nn
from torch import Tensor

from utils.ExpConfigs import ExpConfigs


class Loss(nn.Module):
    def __init__(self, configs:ExpConfigs):
        '''
        Dual loss used in some models, like Ada_MSHyper
        '''
        super(Loss, self).__init__()

    def forward(
        self, 
        pred: Tensor, 
        true: Tensor, 
        mask: Tensor | None = None, 
        loss2: Tensor | None = None, 
        **kwargs
    ) -> dict[str, Tensor]:
        # BEGIN adaptor
        if mask is None:
            mask = torch.ones_like(true, device=true.device)
        if loss2 is None:
            raise ValueError
        # END adaptor

        residual = (pred - true) * mask
        num_eval = mask.sum()

        return {
            "loss": (residual ** 2).sum() / (num_eval if num_eval > 0 else 1) + loss2
        }