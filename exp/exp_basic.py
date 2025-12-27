# Code from: https://github.com/Ladbaby/PyOmniTS
import os
from abc import abstractmethod

import torch

from utils.ExpConfigs import ExpConfigs
from utils.globals import logger


class Exp_Basic(object):
    def __init__(self, configs: ExpConfigs):
        self.configs = configs
        self.device = self._acquire_device()

    def _acquire_device(self):
        if self.configs.use_gpu:
            if self.configs.use_multi_gpu:
                # multi GPU
                # configs priority: --gpu_ids > accelerate configs
                if self.configs.gpu_ids is not None:
                    # case 1: --gpu_ids is given
                    device_ids = self.configs.gpu_ids.replace(' ', '').split(',')
                    device_ids = [int(id_) for id_ in device_ids]
                    self.configs.gpu_id = device_ids[0]
                    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids
                    logger.debug(f"CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES', '')}")
                else:
                    # case 2: --devices is not given, using accelerate configs
                    logger.warning(f"Trying to use accelerate's multi gpu settings. Since --gpu_ids argument is not given.")
            else:
                # single GPU
                logger.debug(f"CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES', '')}")
            device = torch.device(f'cuda:{self.configs.gpu_id}')
            logger.debug(f'Primary GPU: {self.configs.gpu_id}')
        else:
            device = torch.device('cpu')
            logger.warning("GPU is not available. Check your PyTorch installation.")
        return device
    
    @abstractmethod
    def _build_model(self):
        ...

    @abstractmethod
    def _get_data(self):
        ...

    @abstractmethod
    def vali(self):
        ...

    @abstractmethod
    def train(self):
        ...

    @abstractmethod
    def test(self):
        ...
