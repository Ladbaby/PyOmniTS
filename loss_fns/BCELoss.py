import torch
import torch.nn as nn

class Loss(nn.Module):
    '''
    Binary Cross Entropy
    '''
    def __init__(self):
        super(Loss, self).__init__()
        self.criterion = nn.BCELoss()

    def forward(self, pred_class, true_class, **kwargs):
        '''
        - pred: [BATCH_SIZE, N_CLASSES] torch.float32
        - true: [BATCH_SIZE, N_CLASSES] torch.float32
        '''
        return {
            "loss": self.criterion(torch.sigmoid(pred_class), true_class)
        }