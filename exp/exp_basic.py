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
        if configs.allow_tf32 and self.device != torch.device('cpu'):
            self._enable_tf32()

    def _acquire_device(self):
        '''
        device managed by accelerate if configs.use_multi_gpu is 1 and configs.gpu_ids is None, otherwise managed by Exp_Basic.
        '''
        if self.configs.use_gpu and torch.cuda.is_available():
            if self.configs.use_multi_gpu:
                # multi GPU
                # configs priority: --gpu_ids > accelerate configs
                if self.configs.gpu_ids is not None:
                    # case 1: --gpu_ids is given
                    logger.warning(f"--gpu_ids is deprecated in this version. Its value will be ignored.")
                # case 2: --devices is not given, using accelerate configs
                logger.warning(f"Trying to use accelerate's multi gpu settings.")
            else:
                # single GPU
                pass
            logger.debug(f"CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES', '')}")
            device = torch.device(f'cuda:{self.configs.gpu_id}')
        else:
            device = torch.device('cpu')
            if self.configs.use_gpu:
                self.configs.use_gpu = 0
                logger.warning("GPU is not available, so --use_gpu is overwritten to 0. Check your PyTorch installation.")
        logger.debug(f'Primary device: {device}')
        return device

    def _enable_tf32(self):
        try: # new api
            torch.backends.fp32_precision = "tf32"
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            torch.backends.cudnn.fp32_precision = "tf32"
            torch.backends.cudnn.conv.fp32_precision = "tf32"
            torch.backends.cudnn.rnn.fp32_precision = "tf32"
        except: # old api
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        def check_tensor_cores(device_id: torch.device):
            major, minor = torch.cuda.get_device_capability(device_id)
            name = torch.cuda.get_device_name(device_id)

            has_tensor_cores = major >= 7  # Volta (7.0+) and newer
            if not has_tensor_cores:
                logger.warning(f"--allow_tf32 1 will not be effective on {device_id} '{name}', since it does not have tensor cores.")
            else:
                logger.debug(f"tf32 available for {device_id}")

        if self.configs.use_multi_gpu:
            device_ids = range(torch.cuda.device_count())
            for device_id in device_ids:
                check_tensor_cores(torch.device(f'cuda:{device_id}'))
        else:
            check_tensor_cores(self.device)
    
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
