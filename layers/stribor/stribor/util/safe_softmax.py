import torch


def safe_softmax(x, dim=-1):
    """
    Same as `torch.softmax` but returns 0 instead of nan when the whole row is -inf.
    The consequence is that not every row is guaranteed to sum up to one.

    Args:
        x (tensor): Input
        dim (int): Along which dim to perform softmax
    """
    x = torch.softmax(x, dim)
    x = torch.nan_to_num(x)
    return x
