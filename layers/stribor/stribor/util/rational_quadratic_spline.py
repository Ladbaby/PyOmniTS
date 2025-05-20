import layers.stribor.stribor as st
import torch
from torch.nn import functional as F
import numpy as np


def quadratic_spline_latent_dim(dim, n_bins):
    return dim * (3 * n_bins - 1)


def unconstrained_rational_quadratic_spline(inputs, unnorm_widths,
    unnorm_heights, unnorm_derivatives, inverse=False, lower=-1.0, upper=
    1.0, left=None, right=None, bottom=None, top=None, min_bin_width=0.001,
    min_bin_height=0.001, min_derivative=0.001):
    """
    Takes inputs and unnormalized parameters for width, height and
    derivatives of spline bins. Normalizes parameters and applies quadratic spline.
    The domain and codomain can be defined with (lower, upper) or the domain can be
    defined with (left, right) and codomain with (bottom, top).
    Code adapted from https://github.com/bayesiains/nsf

    Args:
        inputs (tensor): Input with shape (..., dim)
        unnorm_widths (tensor): Bin widths (x-axis) with shape (..., dim, n_bins)
        unnorm_heights (tensor): Bin heights (y-axis) between knots with shape (..., dim, n_bins)
        unnorm_derivatives (tensor): Derivatives in knots with shape (..., dim, n_bins - 1) or
            (..., dim, n_bins + 1) (whether derivative in bounds is specified)
        inverse (bool, optional): Whether to invert the calculation. Default: False
        lower (float, optional): Lower domain/codomain bound. Default: -1.
        upper (float, optional): Upper domain/codomain bound. Default: 1.
        left (float or tensor, optional): Lower domain bound. Default: None
        right (float or tensor, optional): Upper domain bound. Default: None
        bottom (float or tensor, optional): Lower codomain bound. Default: None
        top (float or tensor, optional): Upper codomain bound. Default: None
        min_bin_width (float, optional): Minimum bin width. Default: 1e-3
        min_bin_height (float, optional): Minimum bin height. Default: 1e-3
        min_derivative (float, optional): Minimum knot derivative. Default: 1e-3

    Returns:
        outputs (tensor): Spline transformed input (..., dim)
        ljd (tensor): Log-Jacobian diagonal (..., dim)
    """
    if all(x is not None for x in [left, right, top, bottom]):
        if inverse:
            lower = bottom
            upper = top
        else:
            lower = left
            upper = right
    else:
        left = bottom = lower
        right = top = upper
    unnorm_widths = unnorm_widths.expand(*inputs.shape, -1)
    unnorm_heights = unnorm_heights.expand(*inputs.shape, -1)
    unnorm_derivatives = unnorm_derivatives.expand(*inputs.shape, -1)
    inside_interval = (inputs >= lower) & (inputs <= upper)
    outside_interval = ~inside_interval
    outputs = torch.zeros_like(inputs)
    ljd = torch.zeros_like(inputs)
    if unnorm_derivatives.shape[-1] == unnorm_widths.shape[-1] - 1:
        unnorm_derivatives = F.pad(unnorm_derivatives, pad=(1, 1))
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnorm_derivatives[..., 0] = constant
        unnorm_derivatives[..., -1] = constant
    outputs[outside_interval] = inputs[outside_interval]
    ljd[outside_interval] = 0
    if not inside_interval.any():
        return outputs, ljd
    num_bins = unnorm_widths.shape[-1]
    if min_bin_width * num_bins > 1.0:
        raise ValueError('Minimal bin width too large for the number of bins')
    if min_bin_height * num_bins > 1.0:
        raise ValueError('Minimal bin height too large for the number of bins')
    widths = F.softmax(unnorm_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    heights = F.softmax(unnorm_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    derivatives = min_derivative + F.softplus(unnorm_derivatives)
    outputs, ljd = rational_quadratic_spline(inputs=inputs, widths=widths,
        heights=heights, derivatives=derivatives, inside_interval=
        inside_interval, initial_outputs=outputs, initial_ljd=ljd, inverse=
        inverse, left=left, right=right, bottom=bottom, top=top)
    return outputs, ljd


def rational_quadratic_spline(inputs, widths, heights, derivatives,
    inside_interval, initial_outputs, initial_ljd, left, right, bottom, top,
    inverse=False):
    """
    Args:
        inputs (tensor): Input with shape (..., dim)
        widths (tensor): Bin widths with shape (..., dim, n_bins)
        heights (tensor): Bin heights with shape (..., dim, n_bins)
        derivatives (tensor): Derivatives with shape (..., dim, n_bins + 1)
        inside_interval (tensor): Boolean mask, (..., dim)
        initial_outputs: Initial input, usually initialized to zero with shape (..., dim)
        initial_ljd: Same as initial_outputs
        left (float or tensor): Left boundary, if tensor, has shape (..., dim)
        right (float or tensor): Right boundary, same type as left boundary
        bottom (float or tensor): Bottom boundary, same type as left boundary
        top (float or tensor): Top boundary, same type as left boundary
        inverse (bool, optional): Whether to do inverse calculation. Default: False

    Returns:
        outputs: (..., dim)
        ljd: (..., dim)
    """
    inputs = inputs[inside_interval]
    widths = widths[inside_interval, :]
    heights = heights[inside_interval, :]
    derivatives = derivatives[inside_interval, :]

    def boundary_to_tensor(b):
        return b[inside_interval].unsqueeze(-1) if torch.is_tensor(b
            ) else torch.ones(inputs.shape[0], 1) * b
    left = boundary_to_tensor(left)
    right = boundary_to_tensor(right)
    top = boundary_to_tensor(top)
    bottom = boundary_to_tensor(bottom)
    if inverse and ((inputs < bottom).any() or (inputs > top).any()):
        raise ValueError('Inverse input is outside of domain')
    elif not inverse and ((inputs < left).any() or (inputs > right).any()):
        raise ValueError('Input is outside of domain')
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0, None] = left
    cumwidths[..., -1, None] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0, None] = bottom
    cumheights[..., -1, None] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]
    if inverse:
        bin_idx = st.util.searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = st.util.searchsorted(cumwidths, inputs)[..., None]
    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]
    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]
    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[
        ..., 0]
    input_heights = heights.gather(-1, bin_idx)[..., 0]
    if inverse:
        a = (inputs - input_cumheights) * (input_derivatives +
            input_derivatives_plus_one - 2 * input_delta) + input_heights * (
            input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta)
        c = -input_delta * (inputs - input_cumheights)
        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()
        root = 2 * c / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths
        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (input_derivatives +
            input_derivatives_plus_one - 2 * input_delta
            ) * theta_one_minus_theta
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2) + 2 * input_delta *
            theta_one_minus_theta + input_derivatives * (1 - root).pow(2))
        ljd = -torch.log(derivative_numerator) + 2 * torch.log(denominator)
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)
        numerator = input_heights * (input_delta * theta.pow(2) + 
            input_derivatives * theta_one_minus_theta)
        denominator = input_delta + (input_derivatives +
            input_derivatives_plus_one - 2 * input_delta
            ) * theta_one_minus_theta
        outputs = input_cumheights + numerator / denominator
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2) + 2 * input_delta *
            theta_one_minus_theta + input_derivatives * (1 - theta).pow(2))
        ljd = torch.log(derivative_numerator) - 2 * torch.log(denominator)
    initial_outputs[inside_interval], initial_ljd[inside_interval
        ] = outputs, ljd
    return initial_outputs, initial_ljd
