import unittest

import torch
from torch import Tensor

from loss_fns.MSE import Loss
from utils.configs import get_configs


class TestMSE(unittest.TestCase):

    def test_loss_fn(self):
        configs = get_configs(args=[])

        loss_fn = Loss(configs)
        pred = torch.randn(configs.batch_size, configs.pred_len, configs.enc_in)
        true = torch.randn(configs.batch_size, configs.pred_len, configs.enc_in)

        loss_dict: dict[Tensor] = loss_fn(**{
            "pred": pred,
            "true": true
        })
        self.assertEqual(loss_dict["loss"].dim(), 0)  # scalar
