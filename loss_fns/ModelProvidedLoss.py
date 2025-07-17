import torch.nn as nn

from utils.globals import logger
from utils.ExpConfigs import ExpConfigs

class Loss(nn.Module):
    '''
    If your model's loss is not applicable to any of the other models, and you what to calculate the loss inside the model, then this loss is for you.
    e.g., Latent_ODE
    '''
    def __init__(self, configs:ExpConfigs):
        super(Loss, self).__init__()

    def forward(self, loss=None, **kwargs):
        if loss is None:
            logger.exception(f"ModelProvidedLoss expects key 'loss' in model's returned dictionary. The model is supposed to calculate the loss itself in forward function", stack_info=True)

        return {
            "loss": loss
        }