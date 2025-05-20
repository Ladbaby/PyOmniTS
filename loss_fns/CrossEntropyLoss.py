import torch.nn as nn

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, pred_class, true_class, **kwargs):
        '''
        - pred: [BATCH_SIZE, N_CLASSES] torch.float32
        - true: [BATCH_SIZE] LongTensor, which means dtype of torch.int64
        '''
        return {
            "loss": self.criterion(pred_class, true_class)
        }