import torch
import torch.nn as nn

from utils.ExpConfigs import ExpConfigs

class Loss(nn.Module):
    def __init__(self, configs:ExpConfigs):
        super(Loss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, pred_class, true_class, **kwargs):
        '''
        - pred_class: [BATCH_SIZE, N_CLASSES] torch.float32
        - true_class: [BATCH_SIZE, N_CLASSES] torch.float32
            should be converted to LongTensor of shape [BATCH_SIZE], which means dtype of torch.int64
        '''
        return {
            "loss": self.criterion(pred_class, torch.argmax(true_class, dim=1).to(pred_class.device))
        }