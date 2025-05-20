
import torch
import torch.nn.functional as F
from layers.stribor.stribor import ElementwiseTransform


class Sigmoid(ElementwiseTransform):
    """
    Sigmoid transformation.

    Code adapted from torch.distributions.transforms
    """

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, **kwargs):
        finfo = torch.finfo(x.dtype)
        y = torch.clamp(torch.sigmoid(x), min=finfo.tiny, max=1.0 - finfo.eps)
        return y

    def inverse(self, y, **kwargs):
        finfo = torch.finfo(y.dtype)
        y = y.clamp(min=finfo.tiny, max=1.0 - finfo.eps)
        x = y.log() - (-y).log1p()
        return x

    def log_det_jacobian(self, x, y, **kwargs):
        return self.log_diag_jacobian(x, y, **kwargs).sum(-1, keepdim=True)

    def log_diag_jacobian(self, x, y, **kwargs):
        return -F.softplus(-x) - F.softplus(x)


class Logit(Sigmoid):
    """
    Logit transformation. Inverse of sigmoid.
    """

    def forward(self, x, **kwargs):
        return super().inverse(x, **kwargs)

    def inverse(self, y, **kwargs):
        return super().forward(y, **kwargs)

    def log_diag_jacobian(self, x, y, **kwargs):
        return -super().log_diag_jacobian(y, x)
