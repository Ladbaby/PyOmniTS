import torch
import torch.nn as nn

from utils.ExpConfigs import ExpConfigs

class Loss(nn.Module):
    def __init__(self, configs:ExpConfigs):
        '''
        Dual loss used in some models, like Ada_MSHyper
        '''
        super(Loss, self).__init__()

    def forward(self, pred, true, mask=None, loss2=None, **kwargs):
        if mask is None:
            mask = torch.ones_like(true, device=true.device)
        if loss2 is None:
            raise ValueError

        residual = (pred - true) * mask
        num_eval = mask.sum()

        return {
            "loss": (residual ** 2).sum() / (num_eval if num_eval > 0 else 1) + loss2
        }