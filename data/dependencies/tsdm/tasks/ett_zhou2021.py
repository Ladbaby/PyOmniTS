r"""Predicting transformer oil temperatures."""

__all__ = [
    # Classes
    "ETT_Zhou2021",
]


from collections.abc import Callable, Mapping, Sequence
from functools import cached_property
from typing import Any, Literal

from pandas import DataFrame
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from data.dependencies.tsdm.datasets import ETT
from data.dependencies.tsdm.encoders import (
    ChainedEncoder,
    DataFrameEncoder,
    DateTimeEncoder,
    FloatEncoder,
    MinMaxScaler,
    ModularEncoder,
    Standardizer,
    TensorEncoder,
)
from data.dependencies.tsdm.random.samplers import SequenceSampler
from data.dependencies.tsdm.tasks.base import BaseTask


class ETT_Zhou2021(BaseTask):
    r"""Forecasting Oil Temperature on the Electrical-Transformer dataset.

    Paper
    -----

    - | Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
      | Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, Wancai Zhang
      | https://ojs.aaai.org/index.php/AAAI/article/view/17325

    Evaluation Protocol
    -------------------

        ETT (Electricity Transformer Temperature)2: The ETT is a crucial indicator in
        the electric power long-term deployment. We collected 2-year data from two
        separated counties in China. To explore the granularity on the LSTF problem,
        we create separate dataset as {ETTh1, ETTh2}for 1-hour-level and ETTm1 for
        15-minute-level. Each data point consists of the target value ”oil temperature”
        and 6 power load features. The train/val/test is 12/4/4 months

        **Setup:** The input of each dataset is zero-mean normalized.

        For all methods, the input length of recurrent component is chosen from
        {24, 48, 96, 168, 336, 720} for the ETTh1, ETTh2, Weather and Electricity
        dataset, and chosen from {24, 48, 96, 192, 288, 672} for the ETTm dataset.

        The length of preprocessor’s input sequence and decoder’s start token is chosen from
        {24, 48, 96, 168, 336, 480, 720} for the ETTh1, ETTh2, Weather and ECL dataset,
        and {24, 48, 96, 192, 288, 480, 672}for the ETTm dataset.

        In the experiment, the decoder’s start token is a segment truncated from the
        preprocessor’s input sequence, so the length of decoder’s start token must be less
        than the length of preprocessor’s input.

        Appendix E
        [...]
        All the dataset are performed standardization such that the mean of variable
        is 0 and the standard deviation is 1.

    **Forecasting Horizon:** {1d, 2d, 7d, 14d, 30d, 40d}
    **Observation Horizon:**
    **Input_Length**: {24, 48, 96, 168, 336, 720}

    Test-Metric
    -----------

    - MSE: :math:`⅟ₙ∑_{i=1}^{n} | y - ŷ |^2`
    - MAE: :math:`⅟ₙ∑_{i=1}^{n} | y - ŷ |`

    Results
    -------

    TODO: add results
    """

    KeyType = Literal["train", "test", "valid", "joint", "trial", "whole"]
    r"""Type Hint for index."""
    index: Sequence[KeyType] = ["train", "test", "valid", "joint", "trial"]
    r"""Available index."""
    accumulation_function: Callable[..., Tensor]
    r"""Accumulates residuals into loss - usually mean or sum."""

    train_batch_size: int = 32
    r"""Default batch size."""
    eval_batch_size: int = 128
    r"""Default batch size when evaluating."""

    # additional attributes
    preprocessor: ModularEncoder
    r"""Encoder for the observations."""
    observation_horizon: Literal[24, 48, 96, 168, 336, 720] = 96
    r"""The number of datapoints observed during prediction."""
    forecasting_horizon: Literal[24, 48, 168, 336, 960] = 24
    r"""The number of datapoints the model should forecast."""
    TARGET = Literal["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    r"""Type hint available targets."""
    target: TARGET = "OT"
    r"""One of "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"."""

    def __init__(
        self,
        dataset_id: Literal["ETTh1", "ETTh2", "ETTm1", "ETTm2"],
        *,
        # dataset_id: Literal["ETTh1", "ETTh2", "ETTm1", "ETTm2"] = "ETTh1",
        forecasting_horizon: Literal[24, 48, 168, 336, 960] = 24,
        observation_horizon: Literal[24, 48, 96, 168, 336, 720] = 96,
        target: TARGET = "OT",
        eval_batch_size: int = 128,
        train_batch_size: int = 32,
        test_metric: Literal["MSE", "MAE"] = "MSE",
    ):
        super().__init__()
        self.target = target
        self.forecasting_horizon = forecasting_horizon
        self.observation_horizon = observation_horizon
        self.eval_batch_size = eval_batch_size
        self.train_batch_size = train_batch_size

        self.dataset_id = dataset_id
        self.dataset.name = dataset_id

        self.horizon = self.observation_horizon + self.forecasting_horizon
        self.accumulation_function = nn.Identity()

        self.preprocessor = ChainedEncoder(
            TensorEncoder(),
            DataFrameEncoder(
                Standardizer() @ FloatEncoder(),
                index_encoders=MinMaxScaler() @ DateTimeEncoder(),
            ),
        )

        # Fit the Preprocessors
        self.preprocessor.fit(self.splits["train"])
        # Set the Encoder
        # self.pre_encoder = initialize_from(ENCODERS, __name__=pre_encoder)

    @cached_property
    def dataset(self) -> DataFrame:
        r"""Return the dataset."""
        return ETT()[self.dataset_id]

    @cached_property
    def test_metric(self) -> Callable[[Tensor, Tensor], Tensor]:
        r"""The test metric."""
        return nn.MSELoss()

    @cached_property
    def splits(self) -> Mapping[KeyType, DataFrame]:
        r"""Split the dataset into train, test and validation."""
        _splits: dict[Any, DataFrame] = {
            "train": self.dataset.loc["2016-07-01":"2017-06-30"],  # type: ignore[misc]
            "valid": self.dataset.loc["2017-07-01":"2017-10-31"],  # type: ignore[misc]
            "joint": self.dataset.loc["2016-07-01":"2017-10-31"],  # type: ignore[misc]
            "trial": self.dataset.loc["2017-11-01":"2018-02-28"],  # type: ignore[misc]
            "whole": self.dataset,
        }
        _splits["test"] = _splits["trial"]  # alias
        return _splits

    def get_dataloader(
        self,
        key: KeyType,
        /,
        shuffle: bool = True,
        **kwargs: Any,
    ) -> DataLoader:
        r"""Return a DataLoader for the training-dataset with the given batch_size.

        If encode=True, then it will create a dataloader with two outputs

        (inputs, targets)

        where inputs = pre_encoder.encode(masked_batch).

        Parameters
        ----------
        key: Literal["train", "valid", "test"]
            Dataset part from which to construct the DataLoader
        shuffle: bool = True

        Returns
        -------
        DataLoader
        """
        if key == "test":
            assert not shuffle, "Don't shuffle when evaluating test-dataset!"
        if key == "test" and "drop_last" in kwargs:
            assert not kwargs["drop_last"], "Don't drop when evaluating test-dataset!"

        ds = self.splits[key]
        tensors = self.preprocessor.transform(ds)
        # tensors = self.encoder.encode(ds)
        dataset = TensorDataset(*tensors)
        sampler = SequenceSampler(
            dataset, seq_len=self.horizon, stride=1, shuffle=shuffle  # type: ignore[arg-type]
        )

        return DataLoader(dataset, sampler=sampler, **kwargs)
