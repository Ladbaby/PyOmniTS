import importlib
from copy import deepcopy

import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange

from models.VGGish import Model as VGGish
from utils.ExpConfigs import ExpConfigs
from utils.globals import logger

class Model(nn.Module):
    """
    Adaptor model that calculate VGGish feature x_repr of shape [BATCH_SIZE, SEQ_LEN, D_MODEL] for input x, then invoke backbone model.
    """
    def __init__(self, configs: ExpConfigs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.configs = configs
        self.seq_len = configs.seq_len_max_irr or configs.seq_len # equal to seq_len_max_irr if not None, else seq_len
        self.pred_len = configs.pred_len_max_irr or configs.pred_len

        assert self.task_name in ["classification", "representation_learning"], "_OpenMIC_Adaptor only suppports '--task_name classification' or 'representation_learning'"

        configs_vggish = deepcopy(self.configs)
        configs_vggish.task_name = "representation_learning"
        configs_vggish.is_training = 0
        self.vggish = VGGish(configs_vggish)

        # dynamically import the desired model class
        configs_backbone = deepcopy(self.configs)
        configs_backbone.seq_len = 10
        model_module = importlib.import_module("models." + configs.model_name)
        self.backbone = model_module.Model(configs_backbone)

    def forward(
        self, 
        **kwargs
    ):
        # add x_repr as input
        if "x_repr" not in kwargs.keys():
            kwargs["x_repr"] = self.vggish(**kwargs)["pred_repr"]
            kwargs["x"] = kwargs["x_repr"]
        if self.configs.task_name == "classification":
            return self.backbone(**kwargs)
        else:
            raise NotImplementedError
