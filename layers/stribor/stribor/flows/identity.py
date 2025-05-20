
import torch
from layers.stribor.stribor import ElementwiseTransform


class Identity(ElementwiseTransform):
    """
    Identity transformation.
    Doesn't change the input, log-Jacobian is 0.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, **kwargs):
        return x

    def inverse(self, y, **kwargs):
        return y

    def log_det_jacobian(self, x, y, **kwargs):
        return torch.zeros_like(x[..., :1])

    def log_diag_jacobian(self, x, y, **kwargs):
        return torch.zeros_like(x)
