# Code from: https://github.com/Ladbaby/PyOmniTS
import unittest

import torch
from torch import Tensor

from models.GRU_D import Model
from utils.configs import get_configs


class TestGRU_D(unittest.TestCase):

    def test_model(self):
        configs = get_configs(args=[])
        model = Model(configs)

        x = torch.randn(configs.batch_size, configs.seq_len, configs.enc_in)
        result_dict: dict[Tensor] = model(**{"x": x})

        self.assertEqual(result_dict["pred"].shape, torch.Size((configs.batch_size, configs.pred_len, configs.c_out)))
        self.assertEqual(result_dict["true"].shape, torch.Size((configs.batch_size, configs.pred_len, configs.c_out)))
