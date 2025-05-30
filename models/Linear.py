import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange

from utils.ExpConfigs import ExpConfigs
from utils.globals import logger

class Model(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, configs: ExpConfigs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.configs = configs
        self.seq_len = configs.seq_len_max_irr or configs.seq_len # equal to seq_len_max_irr if not None, else seq_len
        self.pred_len = configs.pred_len_max_irr or configs.pred_len
        if configs.task_name in ["short_term_forecast", "long_term_forecast"]:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)
            # Use this line if you want to visualize the weights
            self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        elif configs.task_name == "classification":
            self.decoder_classification = nn.Linear(self.seq_len * configs.enc_in, configs.n_classes)
            self.decoder_classification_repr = nn.Linear(self.seq_len * configs.d_model, configs.n_classes)
        else: 
            raise NotImplementedError
        self.initialize_weights()

    def forward(
        self, 
        x: Tensor,
        x_repr: Tensor = None,
        y: Tensor = None,
        y_mask: Tensor = None,
        y_class: Tensor = None,
        **kwargs
    ):
        # BEGIN adaptor
        BATCH_SIZE, SEQ_LEN, ENC_IN = x.shape
        Y_LEN = self.pred_len
        if y is None:
            if self.configs.task_name in ["short_term_forecast", "long_term_forecast"]:
                logger.warning(f"y is missing for the model input. This is only reasonable when the model is testing flops!")
            y = torch.ones((BATCH_SIZE, Y_LEN, ENC_IN), dtype=x.dtype, device=x.device)
        if y_mask is None:
            y_mask = torch.ones_like(y, device=y.device, dtype=y.dtype)
        if y_class is None:
            if self.configs.task_name == "classification":
                logger.warning(f"y_class is missing for the model input. This is only reasonable when the model is testing flops!")
            y_class = torch.ones((BATCH_SIZE), dtype=x.dtype, device=x.device)
        # END adaptor

        if self.configs.task_name in ['long_term_forecast', 'short_term_forecast']:
            # x: [Batch, Input length, Channel]
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
            f_dim = -1 if self.configs.features == 'MS' else 0
            PRED_LEN = y.shape[1]
            return {
                "pred": x[:, -PRED_LEN:, f_dim:],
                "true": y[:, :, f_dim:],
                "mask": y_mask[:, :, f_dim:]
            }
        elif self.configs.task_name == "classification":
            if x_repr is None:
                return {
                    "pred_class": self.decoder_classification(rearrange(x, "B L N -> B (N L)")),
                    "true_class": y_class
                }
            else:
                return {
                    "pred_class": self.decoder_classification_repr(rearrange(x_repr, "B L N -> B (N L)")),
                    "true_class": y_class
                }
        else:
            raise NotImplementedError


    def initialize_weights(self,):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)