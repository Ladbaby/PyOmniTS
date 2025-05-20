import layers.stribor.stribor as st
import math
import torch
from torch.nn import functional as F
__all__ = ['cubic_spline_latent_dim', 'cubic_spline',
    'unconstrained_cubic_spline']


def cubic_spline_latent_dim(dim, n_bins):
    return dim * (2 * n_bins + 2)


DEFAULT_MIN_BIN_WIDTH = 0.01
DEFAULT_MIN_BIN_HEIGHT = 0.01
DEFAULT_EPS = 1e-05
DEFAULT_QUADRATIC_THRESHOLD = 0.001


def cbrt(x):
    """ Cube root. Equivalent to torch.pow(x, 1/3), but numerically stable """
    return torch.sign(x) * torch.exp(torch.log(torch.abs(x)) / 3.0)


def unconstrained_cubic_spline(inputs, unnormalized_widths,
    unnormalized_heights, unnorm_derivatives, inverse=False, lower=-1,
    upper=1, tails='linear', min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT, eps=DEFAULT_EPS,
    quadratic_threshold=DEFAULT_QUADRATIC_THRESHOLD):
    unnormalized_widths = unnormalized_widths.expand(*inputs.shape, -1)
    unnormalized_heights = unnormalized_heights.expand(*inputs.shape, -1)
    unnorm_derivatives_left = unnorm_derivatives[..., 0, None].expand(*
        inputs.shape, -1)
    unnorm_derivatives_right = unnorm_derivatives[..., 1, None].expand(*
        inputs.shape, -1)
    inside_interval = (inputs >= lower) & (inputs <= upper)
    outside_interval = ~inside_interval
    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)
    if tails == 'linear':
        outputs[outside_interval] = inputs[outside_interval]
        logabsdet[outside_interval] = 0
    else:
        raise RuntimeError('{} tails are not implemented.'.format(tails))
    if not inside_interval.any():
        return outputs, logabsdet
    outputs[inside_interval], logabsdet[inside_interval] = cubic_spline(inputs
        =inputs[inside_interval], unnormalized_widths=unnormalized_widths[
        inside_interval, :], unnormalized_heights=unnormalized_heights[
        inside_interval, :], unnorm_derivatives_left=
        unnorm_derivatives_left[inside_interval, :],
        unnorm_derivatives_right=unnorm_derivatives_right[inside_interval,
        :], inverse=inverse, left=lower, right=upper, bottom=lower, top=
        upper, min_bin_width=min_bin_width, min_bin_height=min_bin_height,
        eps=eps, quadratic_threshold=quadratic_threshold)
    return outputs, logabsdet


def cubic_spline(inputs, unnormalized_widths, unnormalized_heights,
    unnorm_derivatives_left, unnorm_derivatives_right, inverse=False, left=
    0.0, right=1.0, bottom=0.0, top=1.0, min_bin_width=
    DEFAULT_MIN_BIN_WIDTH, min_bin_height=DEFAULT_MIN_BIN_HEIGHT, eps=
    DEFAULT_EPS, quadratic_threshold=DEFAULT_QUADRATIC_THRESHOLD):
    """
    References:
    > Blinn, J. F. (2007). How to solve a cubic equation, part 5: Back to numerics. IEEE Computer
    Graphics and Applications, 27(3):78–89.
    """
    if not inverse and (torch.min(inputs) < left or torch.max(inputs) > right):
        raise ValueError('Input is outside the domain')
    elif inverse and (torch.min(inputs) < bottom or torch.max(inputs) > top):
        raise ValueError('Input is outside the domain')
    num_bins = unnormalized_widths.shape[-1]
    if min_bin_width * num_bins > 1.0:
        raise ValueError('Minimal bin width too large for the number of bins')
    if min_bin_height * num_bins > 1.0:
        raise ValueError('Minimal bin height too large for the number of bins')
    if inverse:
        inputs = (inputs - bottom) / (top - bottom)
    else:
        inputs = (inputs - left) / (right - left)
    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths[..., -1] = 1
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)
    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights[..., -1] = 1
    cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
    slopes = heights / widths
    min_something_1 = torch.min(torch.abs(slopes[..., :-1]), torch.abs(
        slopes[..., 1:]))
    min_something_2 = 0.5 * (widths[..., 1:] * slopes[..., :-1] + widths[
        ..., :-1] * slopes[..., 1:]) / (widths[..., :-1] + widths[..., 1:])
    min_something = torch.min(min_something_1, min_something_2)
    derivatives_left = torch.sigmoid(unnorm_derivatives_left) * 3 * slopes[
        ..., 0][..., None]
    derivatives_right = torch.sigmoid(unnorm_derivatives_right) * 3 * slopes[
        ..., -1][..., None]
    derivatives = min_something * (torch.sign(slopes[..., :-1]) + torch.
        sign(slopes[..., 1:]))
    derivatives = torch.cat([derivatives_left, derivatives,
        derivatives_right], dim=-1)
    a = (derivatives[..., :-1] + derivatives[..., 1:] - 2 * slopes
        ) / widths.pow(2)
    b = (3 * slopes - 2 * derivatives[..., :-1] - derivatives[..., 1:]
        ) / widths
    c = derivatives[..., :-1]
    d = cumheights[..., :-1]
    if inverse:
        bin_idx = st.util.searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = st.util.searchsorted(cumwidths, inputs)[..., None]
    inputs_a = a.gather(-1, bin_idx)[..., 0]
    inputs_b = b.gather(-1, bin_idx)[..., 0]
    inputs_c = c.gather(-1, bin_idx)[..., 0]
    inputs_d = d.gather(-1, bin_idx)[..., 0]
    input_left_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_right_cumwidths = cumwidths.gather(-1, bin_idx + 1)[..., 0]
    if inverse:
        inputs_b_ = inputs_b / inputs_a / 3.0
        inputs_c_ = inputs_c / inputs_a / 3.0
        inputs_d_ = (inputs_d - inputs) / inputs_a
        delta_1 = -inputs_b_.pow(2) + inputs_c_
        delta_2 = -inputs_c_ * inputs_b_ + inputs_d_
        delta_3 = inputs_b_ * inputs_d_ - inputs_c_.pow(2)
        discriminant = 4.0 * delta_1 * delta_3 - delta_2.pow(2)
        depressed_1 = -2.0 * inputs_b_ * delta_1 + delta_2
        depressed_2 = delta_1
        three_roots_mask = discriminant > 0
        one_root_mask = discriminant <= 0
        outputs = torch.zeros_like(inputs)
        p = cbrt((-depressed_1[one_root_mask] + torch.sqrt(-discriminant[
            one_root_mask])) / 2.0)
        q = cbrt((-depressed_1[one_root_mask] - torch.sqrt(-discriminant[
            one_root_mask])) / 2.0)
        outputs[one_root_mask] = p + q - inputs_b_[one_root_mask
            ] + input_left_cumwidths[one_root_mask]
        theta = torch.atan2(torch.sqrt(discriminant[three_roots_mask]), -
            depressed_1[three_roots_mask])
        theta /= 3.0
        cubic_root_1 = torch.cos(theta)
        cubic_root_2 = torch.sin(theta)
        root_1 = cubic_root_1
        root_2 = -0.5 * cubic_root_1 - 0.5 * math.sqrt(3) * cubic_root_2
        root_3 = -0.5 * cubic_root_1 + 0.5 * math.sqrt(3) * cubic_root_2
        root_scale = 2 * torch.sqrt(-depressed_2[three_roots_mask])
        root_shift = -inputs_b_[three_roots_mask] + input_left_cumwidths[
            three_roots_mask]
        root_1 = root_1 * root_scale + root_shift
        root_2 = root_2 * root_scale + root_shift
        root_3 = root_3 * root_scale + root_shift
        root1_mask = (input_left_cumwidths[three_roots_mask] - eps < root_1
            ).float()
        root1_mask *= (root_1 < input_right_cumwidths[three_roots_mask] + eps
            ).float()
        root2_mask = (input_left_cumwidths[three_roots_mask] - eps < root_2
            ).float()
        root2_mask *= (root_2 < input_right_cumwidths[three_roots_mask] + eps
            ).float()
        root3_mask = (input_left_cumwidths[three_roots_mask] - eps < root_3
            ).float()
        root3_mask *= (root_3 < input_right_cumwidths[three_roots_mask] + eps
            ).float()
        roots = torch.stack([root_1, root_2, root_3], dim=-1)
        masks = torch.stack([root1_mask, root2_mask, root3_mask], dim=-1)
        mask_index = torch.argsort(masks, dim=-1, descending=True)[..., 0][
            ..., None]
        outputs[three_roots_mask] = torch.gather(roots, dim=-1, index=
            mask_index).view(-1)
        quadratic_mask = inputs_a.abs() < quadratic_threshold
        a = inputs_b[quadratic_mask]
        b = inputs_c[quadratic_mask]
        c = inputs_d[quadratic_mask] - inputs[quadratic_mask]
        alpha = (-b + torch.sqrt(b.pow(2) - 4 * a * c)) / (2 * a)
        outputs[quadratic_mask] = alpha + input_left_cumwidths[quadratic_mask]
        shifted_outputs = outputs - input_left_cumwidths
        logabsdet = -torch.log(3 * inputs_a * shifted_outputs.pow(2) + 2 *
            inputs_b * shifted_outputs + inputs_c)
    else:
        shifted_inputs = inputs - input_left_cumwidths
        outputs = inputs_a * shifted_inputs.pow(3
            ) + inputs_b * shifted_inputs.pow(2
            ) + inputs_c * shifted_inputs + inputs_d
        logabsdet = torch.log(3 * inputs_a * shifted_inputs.pow(2) + 2 *
            inputs_b * shifted_inputs + inputs_c)
    if inverse:
        outputs = outputs * (right - left) + left
        logabsdet = logabsdet - math.log(top - bottom) + math.log(right - left)
    else:
        outputs = outputs * (top - bottom) + bottom
        logabsdet = logabsdet + math.log(top - bottom) - math.log(right - left)
    return outputs, logabsdet
