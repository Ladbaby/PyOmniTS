# Code from: https://github.com/Ladbaby/PyOmniTS
import unittest

from exp.exp_main import Exp_Main
from utils.configs import get_configs


class TestExpMain(unittest.TestCase):

    def test_exp_main(self):
        configs = get_configs(args=[])
        exp = Exp_Main(configs)
        self.assertIsNotNone(exp.configs)
        self.assertIsNotNone(exp.device)
