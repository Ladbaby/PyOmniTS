# Code from: https://github.com/Ladbaby/PyOmniTS
import unittest

from data.data_provider.datasets.USHCN import Data
from utils.configs import get_configs


class TestUSHCN(unittest.TestCase):

    def test_dataset(self):
        configs = get_configs(args=[
            "--collate_fn", "collate_fn"
        ])

        ds = Data(configs)
        self.assertGreater(len(ds), 0)

        batch_dict = ds[0]
        self.assertEqual(len(batch_dict["x"].shape), 2)
