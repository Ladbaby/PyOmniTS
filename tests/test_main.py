# Code from: https://github.com/Ladbaby/PyOmniTS
import unittest

from main import main
from utils.configs import get_configs


class TestMain(unittest.TestCase):

    def test_main(self):
        configs = get_configs(args=[
            "--train_epochs", "1",
            "--is_training", "1",
            "--itr", "1",
            "--model_name", "DLinear",
            "--model_id", "DLinear"
        ])
        main(configs)
