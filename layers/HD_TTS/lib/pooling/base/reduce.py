from typing import Literal
from typing import Optional

from torch import Tensor, nn

from .select import SelectOutput

ReductionType = Literal["sum", "mean", "min", "max"]


class Reduce(nn.Module):
    r"""The reduction operator from SRC

    .. math::
        \mathrm{sum}(\mathcal{X}) = \sum_{\mathbf{x}_i \in \mathcal{X}}
        \mathbf{x}_i.
    """

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        pass

    def forward(self,
                x: Tensor,
                s: SelectOutput = None,
                *,
                batch: Optional[Tensor] = None,
                num_nodes: Optional[int] = None,
                **kwargs) -> Tensor:
        r"""Implement the Reduce operation.

        Returns:
            The pooled supernode features :math:`\mathbf{X}_{pool}`.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
