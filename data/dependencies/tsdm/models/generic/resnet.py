r"""#TODO add module summary line.

#TODO add module description.
"""

from collections import OrderedDict
from typing import Any, Final, TypeVar

from torch import Tensor, jit, nn

from data.dependencies.tsdm.models.generic.dense import ReverseDense
from data.dependencies.tsdm.utils import deep_dict_update, initialize_from_config
from data.dependencies.tsdm.utils.decorators import autojit


@autojit
class ResNetBlock(nn.Sequential):
    r"""Pre-activation ResNet block.

    References
    ----------
    - | Identity Mappings in Deep Residual Networks
      | Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
      | European Conference on Computer Vision 2016
      | https://link.springer.com/chapter/10.1007/978-3-319-46493-0_38
    """

    HP: Final[dict] = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__module__": __module__,  # type: ignore[name-defined]
        "input_size": None,
        "num_subblocks": 2,
        "subblocks": [
            # {
            #     "__name__": "BatchNorm1d",
            #     "__module__": "torch.nn",
            #     "num_features": int,
            #     "eps": 1e-05,
            #     "momentum": 0.1,
            #     "affine": True,
            #     "track_running_stats": True,
            # },
            ReverseDense.HP,
        ],
    }

    def __init__(self, **HP: Any) -> None:
        super().__init__()

        self.CFG = HP = deep_dict_update(self.HP, HP)

        assert HP["input_size"] is not None, "input_size is required!"

        for layer in HP["subblocks"]:
            if layer["__name__"] == "Linear":
                layer["in_features"] = HP["input_size"]
                layer["out_features"] = HP["input_size"]
            if layer["__name__"] == "BatchNorm1d":
                layer["num_features"] = HP["input_size"]
            else:
                layer["input_size"] = HP["input_size"]
                layer["output_size"] = HP["input_size"]

        subblocks: OrderedDict[str, nn.Module] = OrderedDict()

        for k in range(HP["num_subblocks"]):
            key = f"subblock{k}"
            module = nn.Sequential(
                *[initialize_from_config(layer) for layer in HP["subblocks"]]
            )
            self.add_module(key, module)
            subblocks[key] = module

        # self.subblocks = nn.Sequential(subblocks)
        super().__init__(subblocks)


ResNetType = TypeVar("ResNetType", bound="ResNet")


@autojit
class ResNet(nn.ModuleList):
    r"""A ResNet model."""

    HP: Final[dict] = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__module__": __module__,  # type: ignore[name-defined]
        "input_size": None,
        "num_blocks": 5,
        "blocks": ResNetBlock.HP,
    }

    def __new__(cls, *blocks, **hparams):
        r"""Initialize from hyperparameters."""
        assert len(blocks) ^ len(hparams), "Provide either blocks, or hyperparameters!"

        if hparams:
            return cls.from_hyperparameters(**hparams)

        return super().__new__(cls)

    def __init__(self, *blocks: Any, **hparams: Any) -> None:
        assert len(blocks) ^ len(hparams), "Provide either blocks, or hyperparameters!"

        if hparams:
            return
        super().__init__(*blocks, **hparams)

    @classmethod
    def from_hyperparameters(
        cls: type[ResNetType],
        *,
        input_size: int,
        num_blocks: int = 5,
        block_cfg: dict = ResNetBlock.HP,
    ) -> ResNetType:
        r"""Create a ResNet model from hyperparameters."""
        if "input_size" in block_cfg:
            block_cfg["input_size"] = input_size

        blocks: list[nn.Module] = []
        for _ in range(num_blocks):
            module: nn.Module = initialize_from_config(block_cfg)
            blocks.append(module)
        return cls(*blocks)

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r"""Forward pass.

        Parameters
        ----------
        x: Tensor

        Returns
        -------
        Tensor
        """
        for block in self:
            x = x + block(x)
        return x
