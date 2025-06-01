import torch
import torch.nn as nn
from torch import Tensor
from torchaudio.prototype.pipelines import VGGISH
from einops import rearrange

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
        assert self.configs.is_training == 0, "VGGish only suppports '--is_training 0'. Do not train it!"

        self.bundle = VGGISH
        try:
            self.model = self.bundle.get_model()
        except Exception as e:
            logger.exception(e)
            logger.warning(f"It is possible that downloaded weight for VGGish under `~/.cache/torch/hub/torchaudio/models/` is broken. Try manually remove it then rerun.")
            exit(1)
        self.input_proc = VGGISH.get_input_processor()

    def forward(
        self, 
        x: Tensor,
        **kwargs
    ):
        self.model.eval()

        # Convert to mono if stereo
        x = torch.mean(x, dim=-1) # (BATCH_SIZE, SEQ_LEN)

        output = []
        with torch.no_grad():
            for sample in x:
                # The VGGish model expects input of shape [time_length]
                output.append(self.model(self.input_proc(sample)))
        
        if self.configs.task_name in ["representation_learning"]:
            return {
                "pred_repr": torch.stack(output),
            }
        else:
            raise NotImplementedError