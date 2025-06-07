import torch
import torch.nn as nn
from torch import Tensor
from torchaudio.prototype.pipelines import VGGISH
from einops import rearrange

from layers.VGGish.hubconf import vggish
from layers.VGGish.torchvggish.vggish import Postprocessor
from utils.ExpConfigs import ExpConfigs
from utils.globals import logger

class Model(nn.Module):
    """
    warpper for pretrained pipeline torchaudio.prototype.pipelines.VGGISH
    - paper: "CNN architectures for large-scale audio classification" (ICASSP 2017)
    - paper link: http://ieeexplore.ieee.org/document/7952132/
    """
    def __init__(self, configs: ExpConfigs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.configs = configs

        assert self.task_name in ["representation_learning"], "VGGish only suppports '--task_name representation_learning'"

        self._preprocess = VGGISH.get_input_processor()

        self.pipeline = True

        if self.pipeline:
            self.model = VGGISH.get_model()
            self._postprocess = Postprocessor()
        else:
            self.model = vggish(pretrained=False, preprocess=False)

    def forward(
        self, 
        x: Tensor,
        **kwargs
    ):
        # Convert to mono if stereo
        x = torch.mean(x, dim=-1) # (BATCH_SIZE, SEQ_LEN)

        output = []
        for sample in x:
            # The VGGish model expects input of shape [time_length]
            if self.pipeline:
                output.append(self._postprocess(self.model(self._preprocess(sample))))
            else:
                output.append(self.model(self._preprocess(sample)) / 255.0)
        
        if self.configs.task_name in ["representation_learning"]:
            return {
                "pred_repr_time": torch.stack(output),
            }
        else:
            raise NotImplementedError