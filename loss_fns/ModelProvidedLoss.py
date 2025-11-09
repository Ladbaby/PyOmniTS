# Code from: https://github.com/Ladbaby/PyOmniTS
import torch.nn as nn
from torch import Tensor

from utils.globals import logger
from utils.ExpConfigs import ExpConfigs

class Loss(nn.Module):
    '''
    If your model's loss is not applicable to any of the other models, and you what to calculate the loss inside the model, then this loss is for you.
    e.g., Latent_ODE
    '''
    def __init__(self, configs:ExpConfigs):
        super(Loss, self).__init__()

    def forward(
        self, 
        loss: Tensor | None = None, 
        **kwargs
    ):
        if loss is None:
            logger.exception(f"--loss ModelProvidedLoss expects key 'loss' in the returned dictionary of model's forward(). The model is supposed to calculate the loss itself in forward function", stack_info=True)
            exit(1)

        return {
            "loss": loss
        }