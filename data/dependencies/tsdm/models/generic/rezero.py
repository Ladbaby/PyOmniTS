r"""#TODO add module summary line.

#TODO add module description.
"""

__all__ = [
    "ReZero",
    "ReZeroMLP",
    "ResNetBlock",
]

from collections import OrderedDict
from math import ceil, log2
from typing import Any, Final, Optional

import torch
from torch import Tensor, jit, nn
from torch._jit_internal import _copy_to_script_wrapper

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

    HP = {
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


@autojit
class ReZero(nn.Sequential):
    r"""A ReZero model."""

    weights: Tensor
    r"""PARAM: The weights of the model."""

    def __init__(self, *blocks: nn.Module, weights: Optional[Tensor] = None) -> None:
        super().__init__(*blocks)
        weights = torch.zeros(len(blocks)) if weights is None else weights
        self.register_parameter("weights", nn.Parameter(weights.to(torch.float)))

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
        for k, block in enumerate(self):
            x = x + self.weights[k] * block(x)
        return x

    @_copy_to_script_wrapper
    def __getitem__(
        self: nn.Sequential, item: int | slice
    ) -> nn.Module | nn.Sequential:
        r"""Get a sub-model."""
        modules: list[nn.Module] = list(self._modules.values())
        if isinstance(item, slice):
            return ReZero(*modules[item], weights=self.weights[item])  # type: ignore[index]
        return modules[item]

    @jit.export
    def __len__(self) -> int:
        r"""Get the number of sub-models."""
        return len(self._modules)


@autojit
class ConcatEmbedding(nn.Module):
    r"""Maps $x ⟼ [x,w]$.

    Attributes
    ----------
    input_size:  int
    hidden_size: int
    pad_size:    int
    padding: Tensor
    """

    HP = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__doc__": __doc__,
        "__module__": __module__,  # type: ignore[name-defined]
        "input_size": int,
        "hidden_size": int,
    }
    r"""Dictionary of Hyperparameters."""

    # Constants
    input_size: Final[int]
    r"""CONST: The dimensionality of the inputs."""
    hidden_size: Final[int]
    r"""CONST: The dimensionality of the outputs."""
    pad_size: Final[int]
    r"""CONST: The size of the padding."""

    # BUFFERS
    scale: Tensor
    r"""BUFFER: The scaling scalar."""

    # Parameters
    padding: Tensor
    r"""PARAM: The padding vector."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        assert (
            input_size <= hidden_size
        ), f"ConcatEmbedding requires {input_size=} < {hidden_size=}!"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.pad_size = hidden_size - input_size
        self.padding = nn.Parameter(torch.randn(self.pad_size))

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., d) -> (..., d+e)``.

        Parameters
        ----------
        x: Tensor, shape=(...,DIM)

        Returns
        -------
        Tensor, shape=(...,LAT)
        """
        shape = list(x.shape[:-1]) + [self.pad_size]
        z = torch.cat([x, self.padding.expand(shape)], dim=-1)
        torch.cuda.synchronize()  # needed when cat holds 0-size tensor
        return z

    @jit.export
    def inverse(self, z: Tensor) -> Tensor:
        r""".. Signature:: ``(..., d+e) -> (..., d)``.

        The reverse of the forward. Satisfies inverse(forward(x)) = x for any input.

        Parameters
        ----------
        z: Tensor, shape=(...,LEN,LAT)

        Returns
        -------
        Tensor, shape=(...,LEN,DIM)
        """
        return z[..., : self.input_size]


@autojit
class ReZeroMLP(nn.Sequential):
    r"""A ReZero based on MLP and Encoder + Decoder."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        latent_size: Optional[int] = None,
        num_blocks: int = 2,
    ) -> None:
        super().__init__()

        latent_size = (
            2 ** ceil(log2(input_size)) if latent_size is None else latent_size
        )

        self.encoder = ConcatEmbedding(input_size, latent_size)

        blocks = [
            nn.Sequential(
                ReverseDense(latent_size, latent_size // 2),
                ReverseDense(latent_size // 2, latent_size),
            )
            for _ in range(num_blocks)
        ]

        # self.encoder = ReverseDense(input_size=input_size, output_size=latent_size)
        self.blocks = ReZero(*blocks)
        self.decoder = ReverseDense(input_size=latent_size, output_size=output_size)

        super().__init__(*[self.encoder, self.blocks, self.decoder])
