import torch
__all__ = ['FuncAndDiagJac']


class FuncAndDiagJac(torch.autograd.Function):
    """
    Given a f: R^d -> R^d, computes both f(x) and the diagonals of the Jacobian of f(x).
    "Neural Networks with Cheap Differential Operators" (https://arxiv.org/abs/1912.03579)
    https://proceedings.neurips.cc/paper/2019/hash/770f8e448d07586afbf77bb59f698587-Abstract.html

    Args:
        exclusive_net (nn.Module): Neural network with hollow Jacobian, i.e. ith input
            does not influence the ith output. For example: `st.net.MADE`
        dimwise_net (nn.Module): Neural network with inputs `(t, x, h)` where x has `(B,1)` shape
            and `h` has `(B,H)`. It applies the f: R^{H+2} -> R function conditioned on `t` and `h`.
        t (tensor): Input time of shape (1,)
        x (tensor): Input of shape (...,D)
        flat_params (): Parameters of the exclusive and dimwise net. Can be obtained with
            `st.util.flatten_params(exclusive_net, dimwise_net)`.
        order (int): The derivation order. Default: 1
    """

    @staticmethod
    def forward(ctx, exclusive_net, dimwise_net, t, x, latent, flat_params,
        order=1):
        ctx.exclusive_net = exclusive_net
        ctx.dimwise_net = dimwise_net
        shape = x.shape
        with torch.enable_grad():
            t = t.detach().requires_grad_(True)
            x = x.detach().requires_grad_(True)
            if latent is not None:
                latent = latent.detach().requires_grad_(True)
            h = exclusive_net(t, x)
            x_ = x.view(-1, 1)
            h = h.contiguous().view(x_.shape[0], -1)
            h_detached = h.clone().detach().requires_grad_(True)
            if latent is not None:
                latent_ = latent.clone().unsqueeze(-2).repeat_interleave(x.
                    shape[-1], dim=-2)
                latent_ = torch.cat([h_detached, latent_.contiguous().view(
                    x_.shape[0], -1)], -1)
            else:
                latent_ = h_detached.clone()
            output = dimwise_net(t, x_, latent=latent_).view(*shape)
            djac = torch.autograd.grad(output.sum(), x, create_graph=True)[0]
            while order > 1:
                djac = torch.autograd.grad(djac.sum(), x, create_graph=True)[0]
                order -= 1
            ctx.save_for_backward(t, x, h, h_detached, output, djac, latent)
            return safe_detach(output), safe_detach(djac)

    @staticmethod
    def backward(ctx, grad_output, grad_djac):
        t, x, h, h_detached, output, djac, latent = ctx.saved_tensors
        grad_t = grad_x = grad_latent = grad_params = None
        f_params = list(ctx.exclusive_net.parameters()) + list(ctx.
            dimwise_net.parameters())
        if latent is not None:
            grad_t, grad_x, grad_h, grad_latent, *grad_params = (torch.
                autograd.grad([output, djac], [t, x, h_detached, latent] +
                f_params, [grad_output, grad_djac], retain_graph=True,
                allow_unused=True))
        else:
            grad_t, grad_x, grad_h, *grad_params = torch.autograd.grad([
                output, djac], [t, x, h_detached] + f_params, [grad_output,
                grad_djac], retain_graph=True, allow_unused=True)
        grad_flat_params = flatten_convert_none_to_zeros(grad_params, f_params)
        if grad_h is not None:
            grad_x_from_h, *grad_params_from_h = torch.autograd.grad(h, [x] +
                f_params, grad_h, retain_graph=True, allow_unused=True)
            grad_x = grad_x + grad_x_from_h
            grad_flat_params = (grad_flat_params +
                flatten_convert_none_to_zeros(grad_params_from_h, f_params))
        return None, None, grad_t, grad_x, grad_latent, grad_flat_params


def safe_detach(tensor):
    return tensor.detach().requires_grad_(tensor.requires_grad)


def flatten_convert_none_to_zeros(sequence, like_sequence):
    flat = [(p.contiguous().view(-1) if p is not None else torch.zeros_like
        (q).view(-1)) for p, q in zip(sequence, like_sequence)]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])
