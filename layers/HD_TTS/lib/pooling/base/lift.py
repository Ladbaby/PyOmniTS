from typing import Optional

from torch import Tensor, nn

from .select import SelectOutput


class Lift(nn.Module):
    r"""The lift operator from SRC

    .. math::
        \mathbf{X}\prime = f(\mathbf{X}_{pool}).
    """

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        pass

    def forward(self,
                x_pool: Tensor,
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
