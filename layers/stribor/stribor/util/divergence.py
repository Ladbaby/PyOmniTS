import numpy as np
import torch
import torch.nn as nn


def divergence_exact(output, input):
    diag = torch.zeros_like(input)
    for i in range(input.shape[-1]):
        diag[..., i] += torch.autograd.grad(output[..., i].sum(), input,
            create_graph=True)[0].contiguous()[..., i].contiguous()
    return diag


def divergence_exact_for_sets(output, input):
    diag = torch.zeros_like(input)
    for i in range(input.shape[-2]):
        for j in range(input.shape[-1]):
            diag[..., i, j] += torch.autograd.grad(output[..., i, j].sum(),
                input, create_graph=True)[0].contiguous()[..., i, j
                ].contiguous()
    return diag


def divergence_approx(output, input, e, samples=1):
    out = 0
    for _ in range(samples):
        out += torch.autograd.grad(output, input, e, create_graph=True)[0
            ] * e / samples
    return out


def divergence_from_jacobian(f, inputs):
    """
    Calculates exact divergence for any input shape.
    Best used for input-output pairs with the same shape.

    Args:
        f (callable): Function that transforms a single input or a tuple
            of inputs to an output of same size
        inputs (tensor, Tuple[tensor]): Inputs to the function
    """
    if not isinstance(inputs, tuple):
        inputs = inputs,
    output = f(*inputs)
    jac = torch.autograd.functional.jacobian(f, inputs, vectorize=True)

    def get_diagonal(jac, x):
        smaller_shape = output if np.prod(output.shape) < np.prod(x.shape
            ) else x
        return torch.diag(jac.view(np.prod(x.shape), -1)).view(*
            smaller_shape.shape)
    div = tuple(get_diagonal(d, x) for d, x in zip(jac, inputs))
    if len(div) == 1:
        div = div[0]
    return div
