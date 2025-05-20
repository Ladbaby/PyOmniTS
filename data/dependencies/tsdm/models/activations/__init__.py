r"""Implementations of activation functions.

Notes
-----
Contains activations in both functional and modular form.
  - See `tsdm.models.activations.functional` for functional implementations.
  - See `tsdm.models.activations.modular` for modular implementations.
"""

__all__ = [
    # Sub-Modules
    "functional",
    "modular",
    # Types
    "FunctionalActivation",
    "ModularActivation",
    "Activation",
    # Constants
    "ACTIVATIONS",
    "FUNCTIONAL_ACTIVATIONS",
    "MODULAR_ACTIVATIONS",
    "TORCH_ACTIVATIONS",
    "TORCH_FUNCTIONAL_ACTIVATIONS",
    "TORCH_MODULAR_ACTIVATIONS",
]

from collections.abc import Callable
from typing import Final, TypeAlias

from torch import Tensor, nn

from data.dependencies.tsdm.models.activations import functional, modular

ModularActivation: TypeAlias = nn.Module
r"""Type hint for activation Functions."""

FunctionalActivation: TypeAlias = Callable[..., Tensor]
r"""Type hint for activation Functions."""

Activation: TypeAlias = FunctionalActivation | ModularActivation
r"""Type hint for activation Functions."""

TORCH_FUNCTIONAL_ACTIVATIONS: Final[dict[str, FunctionalActivation]] = {
    "threshold": nn.functional.threshold,
    # Thresholds each element of the input Tensor.
    "threshold_": nn.functional.threshold_,  # type: ignore[attr-defined]
    # In-place version of threshold().
    "relu": nn.functional.relu,
    # Applies the rectified linear unit function element-wise.
    "relu_": nn.functional.relu_,
    # In-place version of relu().
    "hardtanh": nn.functional.hardtanh,
    # Applies the HardTanh function element-wise.
    "hardtanh_": nn.functional.hardtanh_,
    # In-place version of hardtanh().
    "hardswish": nn.functional.hardswish,
    # Applies the hardswish function, element-wise, as described in the paper:
    "relu6": nn.functional.relu6,
    # Applies the element-wise function `ReLU6(x)=\min(\max(0,x),6)`.
    "elu": nn.functional.elu,
    # Applies element-wise, `ELU(x)=\max(0,x)+\min(0,α⋅(\exp(x)−1))`.
    "elu_": nn.functional.elu_,
    # In-place version of elu().
    "selu": nn.functional.selu,
    # Applies element-wise, `SELU(x)=β⋅(\max(0,x)+\min(0,α⋅(e^x−1)))` with `α≈1.677` and `β≈1.05`.
    "celu": nn.functional.celu,
    # Applies element-wise, `CELU(x)= \max(0,x)+\min(0,α⋅(\exp(x/α)−1)`.
    "leaky_relu": nn.functional.leaky_relu,
    # Applies element-wise, `LeakyReLU(x)=\max(0,x)+negative_slope⋅\min(0,x)`.
    "leaky_relu_": nn.functional.leaky_relu_,
    # In-place version of leaky_relu().
    "prelu": nn.functional.prelu,
    # `PReLU(x)=\max(0,x)+ω⋅\min(0,x)` where ω is a learnable parameter.
    "rrelu": nn.functional.rrelu,
    # Randomized leaky ReLU.
    "rrelu_": nn.functional.rrelu_,
    # In-place version of rrelu().
    "glu": nn.functional.glu,
    # The gated linear unit.
    "gelu": nn.functional.gelu,
    # Applies element-wise the function `GELU(x)=x⋅Φ(x)`.
    "logsigmoid": nn.functional.logsigmoid,
    # Applies element-wise `LogSigmoid(x_i)=\log(1/(1+\exp(−x_i)))`.
    "hardshrink": nn.functional.hardshrink,
    # Applies the hard shrinkage function element-wise.
    "tanhshrink": nn.functional.tanhshrink,
    # Applies element-wise, `Tanhshrink(x)=x−\tanh(x)`.
    "softsign": nn.functional.softsign,
    # Applies element-wise, the function `SoftSign(x)=x/(1+∣x∣)`.
    "softplus": nn.functional.softplus,
    # Applies element-wise, the function `Softplus(x)=1/β⋅\log(1+\exp(β⋅x))`.
    "softmin": nn.functional.softmin,
    # Applies a softmin function.
    "softmax": nn.functional.softmax,
    # Applies a softmax function.
    "softshrink": nn.functional.softshrink,
    # Applies the soft shrinkage function elementwise
    "gumbel_softmax": nn.functional.gumbel_softmax,
    # Samples from the Gumbel-Softmax distribution and optionally discretizes.
    "log_softmax": nn.functional.log_softmax,
    # Applies a softmax followed by a logarithm.
    "tanh": nn.functional.tanh,
    # Applies element-wise, `\tanh(x)=(\exp(x)−\exp(−x))/(\exp(x)+\exp(−x))`.
    "sigmoid": nn.functional.sigmoid,
    # Applies the element-wise function `Sigmoid(x)=1/(1+\exp(−x))`.
    "hardsigmoid": nn.functional.hardsigmoid,
    # Applies the hardsigmoid function element-wise.
    "silu": nn.functional.silu,
    # Applies the Sigmoid Linear Unit (SiLU) function, element-wise.
    "mish": nn.functional.mish,
    # Applies the Mish function, element-wise.
    "batch_norm": nn.functional.batch_norm,
    # Applies Batch Normalization for each channel across a batch of data.
    "group_norm": nn.functional.group_norm,
    # Applies Group Normalization for last certain number of dimensions.
    "instance_norm": nn.functional.instance_norm,
    # Applies Instance Normalization for each channel in each data sample in a batch.
    "layer_norm": nn.functional.layer_norm,
    # Applies Layer Normalization for last certain number of dimensions.
    "local_response_norm": nn.functional.local_response_norm,
    # Applies local response normalization over an input signal composed of several input planes.
    "normalize": nn.functional.normalize,
    # Performs Lp normalization of inputs over specified dimension.
}
r"""Dictionary containing all available functional activations in torch."""

FUNCTIONAL_ACTIVATIONS: Final[dict[str, FunctionalActivation]] = {
    **TORCH_FUNCTIONAL_ACTIVATIONS,
    **{},
}
r"""Dictionary containing all available functional activations."""

TORCH_MODULAR_ACTIVATIONS: Final[dict[str, type[ModularActivation]]] = {
    "AdaptiveLogSoftmaxWithLoss": nn.AdaptiveLogSoftmaxWithLoss,
    "ELU": nn.ELU,
    "Hardshrink": nn.Hardshrink,
    "Hardsigmoid": nn.Hardsigmoid,
    "Hardtanh": nn.Hardtanh,
    "Hardswish": nn.Hardswish,
    "Identity": nn.Identity,
    "LeakyReLU": nn.LeakyReLU,
    "LogSigmoid": nn.LogSigmoid,
    "LogSoftmax": nn.LogSoftmax,
    "MultiheadAttention": nn.MultiheadAttention,
    "PReLU": nn.PReLU,
    "ReLU": nn.ReLU,
    "ReLU6": nn.ReLU6,
    "RReLU": nn.RReLU,
    "SELU": nn.SELU,
    "CELU": nn.CELU,
    "GELU": nn.GELU,
    "Sigmoid": nn.Sigmoid,
    "SiLU": nn.SiLU,
    "Softmax": nn.Softmax,
    "Softmax2d": nn.Softmax2d,
    "Softplus": nn.Softplus,
    "Softshrink": nn.Softshrink,
    "Softsign": nn.Softsign,
    "Tanh": nn.Tanh,
    "Tanhshrink": nn.Tanhshrink,
    "Threshold": nn.Threshold,
}
r"""Dictionary containing all available activations in torch."""

MODULAR_ACTIVATIONS: Final[dict[str, type[ModularActivation]]] = {
    **TORCH_MODULAR_ACTIVATIONS,
    **{},
}
r"""Dictionary containing all available activations."""

TORCH_ACTIVATIONS: Final[dict[str, FunctionalActivation | type[ModularActivation]]] = {
    **TORCH_FUNCTIONAL_ACTIVATIONS,
    **TORCH_MODULAR_ACTIVATIONS,
}
r"""Dictionary containing all available activations."""


ACTIVATIONS: Final[dict[str, FunctionalActivation | type[ModularActivation]]] = {
    **TORCH_ACTIVATIONS,
    **MODULAR_ACTIVATIONS,
    **FUNCTIONAL_ACTIVATIONS,
}
r"""Dictionary containing all available activations."""
