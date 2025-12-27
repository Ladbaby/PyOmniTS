# Code from: https://github.com/Ladbaby/PyOmniTS
import torch
import torch.nn as nn
from torch import Tensor

from utils.ExpConfigs import ExpConfigs
from utils.globals import logger


class Loss(nn.Module):
    def __init__(self, configs:ExpConfigs):
        super(Loss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self, 
        pred_class: Tensor, 
        true_class: Tensor, 
        **kwargs
    ) -> dict[str, Tensor]:
        '''
        - pred_class: [BATCH_SIZE, N_CLASSES] torch.float32
        - true_class: [BATCH_SIZE, N_CLASSES] torch.float32
            should be converted to LongTensor of shape [BATCH_SIZE], which means dtype of torch.int64
        '''
        if pred_class.shape != true_class.shape:
            logger.exception(f"CrossEntropyLoss expects pred_class and true_class in the same shape. Currently, {pred_class.shape=} while {true_class.shape=}. This may be caused by an incorrect setting or usage of --n_classes")
            exit(1)
        loss_reduction_none = self.criterion(pred_class, torch.argmax(true_class, dim=1).to(pred_class.device))
        return {
            "loss": loss_reduction_none.mean(),
            "loss_reduction_none": loss_reduction_none # same shape as pred_class
        }