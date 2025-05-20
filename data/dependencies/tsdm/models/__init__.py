r"""Implementation / loading mechanism for models.

There are two types of models:

- Core models: These consist of only a pytorch/tensorflow/mxnet/jax model class.
- Extended models: These consist of a core model and an encoder.
"""

__all__ = [
    # Sub-Packages
    "activations",
    "generic",
    # Type Hints
    "Model",
    "ModelType",
    # Constants
    "MODELS",
    # Classes
    "BaseModel",
    "ODE_RNN",
    "SetFuncTS",
    "GroupedSetFuncTS",
    # Generic
    "MLP",
    "DeepSet",
    "ScaledDotProductAttention",
]

from typing import Final, TypeAlias

from torch import nn

from data.dependencies.tsdm.models import activations, generic
from data.dependencies.tsdm.models._models import BaseModel
from data.dependencies.tsdm.models.generic import MLP, DeepSet, ScaledDotProductAttention
from data.dependencies.tsdm.models.ode_rnn import ODE_RNN
from data.dependencies.tsdm.models.set_function_for_timeseries import GroupedSetFuncTS, SetFuncTS

Model: TypeAlias = nn.Module
r"""Type hint for models."""

ModelType = type[nn.Module]
r"""Type hint for models."""

# TODO: replace Any with BaseModel class
MODELS: Final[dict[str, ModelType]] = {
    "ODE_RNN": ODE_RNN,
    "SetFuncTS": SetFuncTS,
}
r"""Dictionary of all available models."""
