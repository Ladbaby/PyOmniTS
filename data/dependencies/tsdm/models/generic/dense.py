r"""#TODO add module summary line.

#TODO add module description.
"""

from typing import Any, Final, Optional

from torch import Tensor, jit, nn

from data.dependencies.tsdm.utils import deep_dict_update, initialize_from_config
from data.dependencies.tsdm.utils.decorators import autojit


@autojit
class ReverseDense(nn.Module):
    r"""ReverseDense module $x→A⋅ϕ(x)$."""

    HP: Final[dict] = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__module__": __module__,  # type: ignore[name-defined]
        "input_size": None,
        "output_size": None,
        "bias": True,
        "activation": {
            "__name__": "ReLU",
            "__module__": "torch.nn",
            "inplace": False,
        },
    }
    r"""The hyperparameter dictionary."""

    input_size: Final[int]
    r"""The size of the input."""
    output_size: Final[int]
    r"""The size of the output."""

    # PARAMETERS
    weight: Tensor
    r"""The weight matrix."""
    bias: Optional[Tensor]
    r"""The bias vector."""

    def __init__(self, input_size: int, output_size: int, **HP: Any) -> None:
        super().__init__()
        self.CFG = HP = deep_dict_update(self.HP, HP)

        self.input_size = HP["input_size"] = input_size
        self.output_size = HP["output_size"] = output_size

        self.activation: nn.Module = initialize_from_config(HP["activation"])

        self.linear = nn.Linear(HP["input_size"], HP["output_size"], HP["bias"])
        self.weight = self.linear.weight
        self.bias = self.linear.bias

        activation_name = HP["activation"]["__name__"].lower()
        nn.init.kaiming_uniform_(self.weight, nonlinearity=activation_name)

        if self.bias is not None:
            nn.init.kaiming_uniform_(self.bias[None], nonlinearity=activation_name)

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r"""Forward pass."""
        return self.linear(self.activation(x))
