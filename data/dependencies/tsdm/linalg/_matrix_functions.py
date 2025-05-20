r"""#TODO add module summary line.

#TODO add module description.
"""

__all__ = [
    # Functions
    "closest_diag",
    "closest_orth",
    "closest_skew",
    "closest_symm",
    "col_corr",
    "erank",
    "geometric_mean",
    "logarithmic_norm",
    "reldist",
    "reldist_diag",
    "reldist_orth",
    "reldist_skew",
    "reldist_symm",
    "relerank",
    "row_corr",
    "schatten_norm",
    "spectral_abscissa",
    "spectral_radius",
    "stiffness_ratio",
    "vector_norm",
    "tensor_norm",
    "operator_norm",
]


from typing import List

import torch
from torch import Tensor, jit


@jit.script
def erank(x: Tensor) -> Tensor:
    r"""Compute the effective rank of a matrix.

    .. math:: \operatorname{erank}(A) = e^{H(\tfrac{𝛔}{‖𝛔‖_1})} = ∏ σ_{i}^{-σ_i}

    By definition, the effective rank is equal to the exponential of the entropy of the
    distribution of the singular values.

    .. Signature:: ``(..., m, n) -> ...``

    References
    ----------
    - | `The effective rank: A measure of effective dimensionality
        <https://ieeexplore.ieee.org/document/7098875>`_
      | Olivier Roy, Martin Vetterli
      | `15th European Signal Processing Conference (EUSIPCO), 2007
        <https://ieeexplore.ieee.org/xpl/conhome/7067185/proceeding>`_
    """
    σ = torch.linalg.svdvals(x)
    σ = σ / torch.linalg.norm(σ, ord=1, dim=-1)
    entropy = torch.special.entr(σ).sum(dim=-1)
    return torch.exp(entropy)


@jit.script
def relerank(x: Tensor) -> Tensor:
    r"""Compute the relative effective rank of a matrix.

    This is the effective rank scaled by $\min(m,n)$.

    .. Signature:: ``(..., m, n) -> ...``
    """
    return erank(x) / min(x.shape[-2:])


@jit.script
def col_corr(x: Tensor) -> Tensor:
    r"""Compute average column-wise correlation of a matrix.

    .. Signature:: ``(..., m, n) -> ...``
    """
    _, n = x.shape[-2:]
    u = torch.linalg.norm(x, dim=0)
    xx = torch.einsum("...i, ...j -> ...ij", u, u)
    xtx = torch.einsum("...ik, ...il  -> ...kl", x, x)
    I = torch.eye(n, dtype=x.dtype, device=x.device)
    c = I - xtx / xx
    return c.abs().sum(dim=(-2, -1)) / (n * (n - 1))


@jit.script
def row_corr(x: Tensor) -> Tensor:
    r"""Compute average column-wise correlation of a matrix.

    .. Signature:: ``(..., m, n) -> ...``
    """
    m, _ = x.shape[-2:]
    v = torch.linalg.norm(x, dim=1)
    xx = torch.einsum("...i, ...j -> ...ij", v, v)
    xxt = torch.einsum("...kj, ...lj  -> ...kl", x, x)
    I = torch.eye(m, dtype=x.dtype, device=x.device)
    c = I - xxt / xx
    return c.abs().sum(dim=(-2, -1)) / (m * (m - 1))


@jit.script
def closest_symm(x: Tensor, dim: tuple[int, int] = (-2, -1)) -> Tensor:
    r"""Symmetric part of square matrix.

    .. math:: \argmin_{X: X^⊤ = -X} ‖A-X‖

    .. Signature:: ``(..., n, n) -> (..., n, n)``
    """
    rowdim, coldim = dim
    return (x + x.swapaxes(rowdim, coldim)) / 2


@jit.script
def closest_skew(x: Tensor, dim: tuple[int, int] = (-2, -1)) -> Tensor:
    r"""Skew-Symmetric part of a matrix.

    .. math:: \argmin_{X: X^⊤ = X} ‖A-X‖

    .. Signature:: ``(..., n, n) -> (..., n, n)``
    """
    rowdim, coldim = dim
    return (x - x.swapaxes(rowdim, coldim)) / 2


@jit.script
def closest_orth(x: Tensor) -> Tensor:
    r"""Orthogonal part of a square matrix.

    .. math:: \argmin_{X: XᵀX = 𝕀} ‖A-X‖

    .. Signature:: ``(..., n, n) -> (..., n, n)``
    """
    U, _, Vt = torch.linalg.svd(x, full_matrices=True)
    Q = torch.einsum("...ij, ...jk->...ik", U, Vt)
    return Q


@jit.script
def closest_diag(x: Tensor) -> Tensor:
    r"""Diagonal part of a square matrix.

    .. math:: \argmin_{X: X⊙𝕀 = X} ‖A-X‖

    .. Signature:: ``(..., n, n) -> (..., n, n)``
    """
    d = torch.diagonal(x, dim1=-2, dim2=-1)
    return torch.diag_embed(d)


@jit.script
def reldist(x: Tensor, y: Tensor) -> Tensor:
    r"""Relative distance between two matrices.

    .. math::  ‖x-y‖/‖y‖

    .. Signature:: ``[(..., m, n), (..., m, n)]  -> (..., n, n)``
    """
    r = torch.linalg.matrix_norm(x - y, ord="fro", dim=(-2, -1))
    yy = torch.linalg.matrix_norm(y, ord="fro", dim=(-2, -1))
    zero = torch.tensor(0.0, dtype=torch.float32, device=x.device)
    return torch.where(yy != 0, r / yy, zero)


@jit.script
def reldist_diag(x: Tensor) -> Tensor:
    r"""Compute the relative distance to being a diagonal matrix.

    .. math:: ‖A-X‖/‖A‖  X = \argmin_{X: X⊙𝕀 = X} ‖A-X‖

    .. Signature:: ``(..., n, n) -> ...``
    """
    return reldist(closest_diag(x), x)


@jit.script
def reldist_symm(x: Tensor) -> Tensor:
    r"""Relative magnitude of closest_symm part.

    .. Signature:: ``(..., n, n) -> ...``
    """
    return reldist(closest_symm(x), x)


@jit.script
def reldist_skew(x: Tensor) -> Tensor:
    r"""Relative magnitude of skew-closest_symm part.

    .. Signature:: ``(..., n, n) -> ...``
    """
    return reldist(closest_skew(x), x)


@jit.script
def reldist_orth(x: Tensor) -> Tensor:
    r"""Relative magnitude of orthogonal part.

    .. math:: \min_{Q: Q^⊤Q = 𝕀} ‖A-Q‖/‖A‖

    .. Signature:: ``(..., n, n) -> ...``
    """
    return reldist(closest_orth(x), x)


@jit.script
def stiffness_ratio(x: Tensor) -> Tensor:
    r"""Compute the stiffness ratio of a matrix.

    .. math:: \frac{ | \Re(λ_\max) | }{ | \Re{λ_\min} | }

    Only applicable if $\Re(λ_i)<0$ for all $i$.

    .. Signature:: ``(..., n, n) -> ...``

    References
    ----------
    - | Numerical Methods for Ordinary Differential Systems: The Initial Value Problem
      | J. D. Lambert
      | ISBN: 978-0-471-92990-1
    """
    x = x.to(dtype=torch.complex128)
    λ = torch.linalg.eigvals(x)
    λ = λ.real
    # Get smallest non-zero eigenvalue.
    maxvals = λ.amax(dim=-1)
    minvals = λ.amin(dim=-1)
    return torch.where(maxvals < 0, minvals / maxvals, float("nan"))


@jit.script
def spectral_radius(x: Tensor) -> Tensor:
    r"""Return $\max_i | λ_i | $.

    .. Signature:: ``(..., n, n) -> ...``
    """  # noqa: RST219
    λ = torch.linalg.eigvals(x)
    return λ.abs().amax(dim=-1)


@jit.script
def spectral_abscissa(x: Tensor) -> Tensor:
    r"""Return $\max_i \Re(λ_i)$.

    .. Signature:: ``(..., n, n) -> ...``
    """
    λ = torch.linalg.eigvals(x)
    return λ.real.amax(dim=-1)


@jit.script
def geometric_mean(x: Tensor, dim: int = -1, keepdim: bool = False) -> Tensor:
    r"""Geometric mean of a tensor.

    .. Signature:: ``(..., n) -> (...)``
    """
    return x.log().nanmean(dim=dim, keepdim=keepdim).exp()


@jit.script
def apply_keepdim(x: Tensor, dim: tuple[int, int], keepdim: bool) -> Tensor:
    r"""Insert dimensions in the right places.

    We assume x was some tensor to which a reduction was applied, such that

    1. The affected dims were mapped, in order, to the last dimensions of x.
    2. The reduction was performed over the last dimensions of x.
    3. We now want to insert the dimensions back into x at the right places.
    """
    if not keepdim:
        return x

    # TODO: sort by absolute value.
    rowdim, coldim = dim
    dims = (rowdim, coldim) if abs(rowdim) < abs(coldim) else (coldim, rowdim)
    for d in dims:
        x = x.unsqueeze(d)
    return x


@jit.script
def logarithmic_norm(
    x: Tensor,
    p: float = 2.0,
    dim: tuple[int, int] = (-2, -1),
    keepdim: bool = False,
    scaled: bool = False,
) -> Tensor:
    r"""Compute the logarithmic norm of a matrix.

    .. math:: \lim_{ε→0⁺} \frac{‖𝕀+εA‖_p-1}{ε}

    .. Signature:: ``(..., n, n) -> ...``

    References
    ----------
    - `What Is the Logarithmic Norm? <https://nhigham.com/2022/01/18/what-is-the-logarithmic-norm/>_`
    - | The logarithmic norm. History and modern theory
      | Gustaf Söderlind, BIT Numerical Mathematics, 2006
      | <https://link.springer.com/article/10.1007/s10543-006-0069-9>_
    """
    rowdim, coldim = dim
    rowdim = rowdim % x.ndim
    coldim = coldim % x.ndim
    dim = (rowdim, coldim)
    M, N = x.shape[rowdim], x.shape[coldim]
    assert M == N, "Matrix must be square."

    if p == 2:
        x = closest_symm(x, dim=dim)
        x = x.swapaxes(rowdim, -2).swapaxes(coldim, -1)
        λ = torch.linalg.eigvals(x)
        r = λ.real.amax(dim=-1)
        return apply_keepdim(r, dim, keepdim)
    if p == -2:
        x = closest_symm(x, dim=dim)
        x = x.swapaxes(rowdim, -2).swapaxes(coldim, -1)
        λ = torch.linalg.eigvals(x)
        r = λ.real.amin(dim=-1)
        return apply_keepdim(r, dim, keepdim)

    m = torch.eye(N, dtype=torch.bool, device=x.device)
    x = torch.where(m, x.real, x.abs())

    if scaled:
        shift = int(coldim < rowdim) * (1 - int(keepdim))
        if p == 1:
            x = x.mean(dim=coldim, keepdim=keepdim)
            return x.amax(dim=rowdim - shift, keepdim=keepdim)
        if p == -1:
            x = x.mean(dim=coldim, keepdim=keepdim)
            return x.amin(dim=rowdim - shift, keepdim=keepdim)
        shift = int(rowdim < coldim) * (1 - int(keepdim))
        if p == float("inf"):
            x = x.mean(dim=rowdim, keepdim=keepdim)
            return x.amax(dim=coldim - shift, keepdim=keepdim)
        if p == -float("inf"):
            x = x.mean(dim=rowdim, keepdim=keepdim)
            return x.amin(dim=coldim - shift, keepdim=keepdim)

    shift = int(coldim < rowdim) * (1 - int(keepdim))
    if p == 1:
        x = x.sum(dim=coldim, keepdim=keepdim)
        return x.amax(dim=rowdim - shift, keepdim=keepdim)
    if p == -1:
        x = x.sum(dim=coldim, keepdim=keepdim)
        return x.amin(dim=rowdim - shift, keepdim=keepdim)
    shift = int(rowdim < coldim) * (1 - int(keepdim))
    if p == float("inf"):
        x = x.sum(dim=rowdim, keepdim=keepdim)
        return x.amax(dim=coldim - shift, keepdim=keepdim)
    if p == -float("inf"):
        x = x.sum(dim=rowdim, keepdim=keepdim)
        return x.amin(dim=coldim - shift, keepdim=keepdim)

    raise NotImplementedError("Currently only p=±1,±2,±inf are supported.")


@jit.script
def schatten_norm(
    x: Tensor,
    p: float = 2.0,
    dim: tuple[int, int] = (-2, -1),
    keepdim: bool = False,
    scaled: bool = False,
) -> Tensor:
    r"""Schatten norm $p$-th order.

    .. math::  ‖A‖_p^p ≔ \tr(|A|^p) = ∑_i σ_i^p

    The Schatten norm is equivalent to the vector norm of the singular values.

    - $p=+∞$: Maximum Singular Value, equivalent to spectral norm $‖A‖_2$.
    - $p=2$: Frobius Norm
    - $p=1$: Nuclear Norm
    - $p=0$: Number of non-zero singular values. Equivalent to rank.
    - $p=-1$: Reciprocal sum of singular values.
    - $p=-2$: Reciprocal sum of squared singular values.
    - $p=+∞$: Minimal Singular Value

    .. Signature:: ``(..., n, n) -> ...``

    References
    ----------
    - | Schatten Norms
      | <https://en.wikipedia.org/wiki/Schatten_norms>_
    """
    if not torch.is_floating_point(x):
        x = x.to(dtype=torch.float)

    rowdim, coldim = dim

    x = x.swapaxes(rowdim, -2).swapaxes(coldim, -1)
    σ = torch.linalg.svdvals(x)
    m = σ != 0

    if p == float("+inf"):
        σ = torch.where(m, σ, float("-inf"))
        maxvals = σ.amax(dim=-1)
        maxvals = torch.where(maxvals == float("-inf"), float("nan"), maxvals)
        return apply_keepdim(maxvals, dim, keepdim)
    if p == float("-inf"):
        σ = torch.where(m, σ, float("+inf"))
        minvals = σ.amin(dim=-1)
        minvals = torch.where(minvals == float("+inf"), float("nan"), minvals)
        return apply_keepdim(minvals, dim, keepdim)
    if p == 0:
        if scaled:
            σ = torch.where(m, σ, float("nan"))
            result = geometric_mean(σ, dim=-1)
        else:
            result = m.sum(dim=-1)
        return apply_keepdim(result, dim, keepdim)

    σ = torch.where(m, σ, float("-inf"))
    σ_max = σ.amax(dim=-1)
    σ = torch.where(m, σ, float("+nan"))
    σ = σ / σ_max

    if scaled:
        result = σ.pow(p).nanmean(dim=-1).pow(1 / p)
    else:
        result = σ.pow(p).nansum(dim=-1).pow(1 / p)
    return apply_keepdim(result, dim, keepdim)


@jit.script
def vector_norm(
    x: Tensor,
    p: float = 2.0,
    dim: int = -1,
    keepdim: bool = True,
    scaled: bool = False,
) -> Tensor:
    r"""Vector norm of $p$-th order.

    +--------+-----------------------------------+------------------------------------+
    |        | standard                          | size normalized                    |
    +========+===================================+====================================+
    | $p=+∞$ | maximum value                     | maximum value                      |
    +--------+-----------------------------------+------------------------------------+
    | $p=+2$ | sum of squared values             | mean of squared values             |
    +--------+-----------------------------------+------------------------------------+
    | $p=+1$ | sum of squared values             | mean of squared values             |
    +--------+-----------------------------------+------------------------------------+
    | $p=±0$ | ∞ or sum of non-zero values       | geometric mean of values           |
    +--------+-----------------------------------+------------------------------------+
    | $p=-1$ | reciprocal sum of absolute values | reciprocal mean of absolute values |
    +--------+-----------------------------------+------------------------------------+
    | $p=-2$ | reciprocal sum of squared values  | reciprocal mean of squared values  |
    +--------+-----------------------------------+------------------------------------+
    | $p=-∞$ | minimum value                     | minimum value                      |
    +--------+-----------------------------------+------------------------------------+

    .. Signature:: ``(..., n) -> ...``
    """
    if not torch.is_floating_point(x):
        x = x.to(dtype=torch.float)
    x = x.abs()

    # TODO: deal with nan values

    if p == float("inf"):
        return x.amax(dim=dim, keepdim=keepdim)
    if p == -float("inf"):
        return x.amin(dim=dim, keepdim=keepdim)
    if p == 0:
        if scaled:
            return geometric_mean(x, dim=dim, keepdim=keepdim)
        return (x != 0).sum(dim=dim, keepdim=keepdim)

    x_max = x.amax(dim=dim, keepdim=True)
    x = x / x_max

    if scaled:
        r = x.pow(p).mean(dim=dim, keepdim=keepdim).pow(1 / p)
    else:
        r = x.pow(p).sum(dim=dim, keepdim=keepdim).pow(1 / p)
    return x_max * r


@jit.script
def tensor_norm(
    x: Tensor,
    p: float = 2.0,
    dim: List[int] = (),  # type: ignore[assignment]
    keepdim: bool = True,
    scaled: bool = False,
) -> Tensor:
    r"""Vector norm of $p$-th order.

    +--------+-----------------------------------+------------------------------------+
    |        | standard                          | size normalized                    |
    +========+===================================+====================================+
    | $p=+∞$ | maximum value                     | maximum value                      |
    +--------+-----------------------------------+------------------------------------+
    | $p=+2$ | sum of squared values             | mean of squared values             |
    +--------+-----------------------------------+------------------------------------+
    | $p=+1$ | sum of squared values             | mean of squared values             |
    +--------+-----------------------------------+------------------------------------+
    | $p=±0$ | ∞ or sum of non-zero values       | geometric mean of values           |
    +--------+-----------------------------------+------------------------------------+
    | $p=-1$ | reciprocal sum of absolute values | reciprocal mean of absolute values |
    +--------+-----------------------------------+------------------------------------+
    | $p=-2$ | reciprocal sum of squared values  | reciprocal mean of squared values  |
    +--------+-----------------------------------+------------------------------------+
    | $p=-∞$ | minimum value                     | minimum value                      |
    +--------+-----------------------------------+------------------------------------+

    .. Signature:: ``(..., n) -> ...``
    """
    if not torch.is_floating_point(x):
        x = x.to(dtype=torch.float)
    x = x.abs()

    # TODO: deal with nan values

    if p == float("inf"):
        return x.amax(dim=dim, keepdim=keepdim)
    if p == -float("inf"):
        return x.amin(dim=dim, keepdim=keepdim)
    if p == 0:
        if scaled:
            return x.log().nanmean(dim=dim, keepdim=keepdim).exp()
        return (x != 0).sum(dim=dim, keepdim=keepdim)

    x_max = x.amax(dim=dim, keepdim=True)
    x = x / x_max

    if scaled:
        r = x.pow(p).mean(dim=dim, keepdim=keepdim).pow(1 / p)
    else:
        r = x.pow(p).sum(dim=dim, keepdim=keepdim).pow(1 / p)
    return x_max * r


@jit.script
def operator_norm(
    x: Tensor,
    p: float = 2.0,
    dim: tuple[int, int] = (-2, -1),
    keepdim: bool = True,
    scaled: bool = False,
) -> Tensor:
    r"""Operator norm of $p$-th order.

    +--------+-----------------------------------+------------------------------------+
    |        | standard                          | size normalized                    |
    +========+===================================+====================================+
    | $p=+∞$ | maximum value                     | maximum value                      |
    +--------+-----------------------------------+------------------------------------+
    | $p=+2$ | sum of squared values             | mean of squared values             |
    +--------+-----------------------------------+------------------------------------+
    | $p=+1$ | sum of squared values             | mean of squared values             |
    +--------+-----------------------------------+------------------------------------+
    | $p=±0$ | ∞ or sum of non-zero values       | geometric mean of values           |
    +--------+-----------------------------------+------------------------------------+
    | $p=-1$ | reciprocal sum of absolute values | reciprocal mean of absolute values |
    +--------+-----------------------------------+------------------------------------+
    | $p=-2$ | reciprocal sum of squared values  | reciprocal mean of squared values  |
    +--------+-----------------------------------+------------------------------------+
    | $p=-∞$ | minimum value                     | minimum value                      |
    +--------+-----------------------------------+------------------------------------+

    .. Signature:: ``(..., n) -> ...``
    """
    rowdim, coldim = dim
    assert x.shape[rowdim] == x.shape[coldim], "Matrix must be square."

    if p == 2:
        x = x.swapaxes(rowdim, -2).swapaxes(coldim, -1)
        σ = torch.linalg.svdvals(x)
        r = σ.amax(dim=-1)
        return apply_keepdim(r, dim, keepdim)
    if p == -2:
        x = x.swapaxes(rowdim, -2).swapaxes(coldim, -1)
        σ = torch.linalg.svdvals(x)
        r = σ.amin(dim=-1)
        return apply_keepdim(r, dim, keepdim)

    x = x.abs()
    shift = int(coldim < rowdim) * int(keepdim)

    if scaled:
        if p == 1:
            x = x.mean(dim=coldim, keepdim=keepdim)
            return x.amax(dim=rowdim - shift, keepdim=keepdim)
        if p == -1:
            x = x.mean(dim=coldim, keepdim=keepdim)
            return x.amin(dim=rowdim - shift, keepdim=keepdim)
        if p == float("inf"):
            x = x.mean(dim=rowdim, keepdim=keepdim)
            return x.amax(dim=coldim + shift, keepdim=keepdim)
        if p == -float("inf"):
            x = x.mean(dim=rowdim, keepdim=keepdim)
            return x.amin(dim=coldim + shift, keepdim=keepdim)

    if p == 1:
        x = x.sum(dim=coldim, keepdim=keepdim)
        return x.amax(dim=rowdim - shift, keepdim=keepdim)
    if p == -1:
        x = x.sum(dim=coldim, keepdim=keepdim)
        return x.amin(dim=rowdim - shift, keepdim=keepdim)
    if p == float("inf"):
        x = x.sum(dim=rowdim, keepdim=keepdim)
        return x.amax(dim=coldim + shift, keepdim=keepdim)
    if p == -float("inf"):
        x = x.sum(dim=rowdim, keepdim=keepdim)
        return x.amin(dim=coldim + shift, keepdim=keepdim)

    raise NotImplementedError("Currently only p=±1,±2,±inf are supported.")
