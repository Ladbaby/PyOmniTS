# Stribor

Normalizing flows and neural flows for PyTorch.

- Normalizing flow defines a complicated probability density function as a transformation of the random variable.
- Neural flow defines continuous time dynamics with invertible neural networks.

## Install package and dependencies

```
pip install stribor
```

## Normalizing flows

### Base densities

- Normal `st.Normal` and `st.UnitNormal` and `st.MultivariateNormal`
- Uniform `st.UnitUniform`
- Or, use distributions from `torch.distributions`

### Invertible transformations

- Activation functions
    - ELU `st.ELU`
    - Leaky ReLU `st.LeakyReLU`
    - Sigmoid `st.Sigmoid`
    - Logit (inverse sigmoid) `st.Logit`
- Affine
    - Element-wise transformation `st.Affine`
    - Linear layer with LU factorization `st.AffineLU`
    - Matrix exponential `st.MatrixExponential`
- Coupling layer that can be combined with any element-wise transformation `st.Coupling`
- Continuous normalizing flows `st.ContinuousTransform`
    - Differential equations with stochastic trace estimation:
        - `st.net.DiffeqMLP`
        - `st.net.DiffeqDeepset`
        - `st.net.DiffeqSelfAttention`
    - Differential equations with fixed zero trace:
        - `st.net.DiffeqZeroTraceMLP`
        - `st.net.DiffeqZeroTraceDeepSet`
        - `st.net.DiffeqZeroTraceAttention`
    - Differential equations with exact trace computation:
        - `st.net.DiffeqExactTraceMLP`
        - `st.net.DiffeqExactTraceDeepSet`
        - `st.net.DiffeqExactTraceAttention`
- Cummulative sum `st.Cumsum` and difference `st.Diff`
    - Across single column `st.CumsumColumn` and `st.DiffColumn`
- Permutations
    - Flipping the indices `st.Flip`
    - Random permutation of indices `st.Permute`
- Spline (quadratic or cubic) element-wise transformation `st.Spline`


### Example: Normalizing flow

To define a normalizing flow, define a base distribution and a series of transformations, e.g.:
```py
import layers.stribor.stribor as st
import torch

dim = 2

base_dist = st.UnitNormal(dim)

transforms = [
    st.Coupling(
        transform=st.Affine(dim, latent_net=st.net.MLP(dim, [64], 2 * dim)),
        mask='ordered_right_half',
    ),
    st.ContinuousTransform(
        dim,
        net=st.net.DiffeqMLP(dim + 1, [64], dim),
    )
]

flow = st.NormalizingFlow(base_dist, transforms)

x = torch.rand(10, dim)
y = flow(x) # Forward transformation
log_prob = flow.log_prob(y) # Log-probability p(y)
```

### Example: Neural flow

Neural flows are defined similarly but now we don't need the base density and all the invertible transformations must depend on time. In particular, at `t=0`, the transformation becomes an identity.
```py
import torch
import layers.stribor.stribor as st

dim = 2

f = st.NeuralFlow([
    st.ContinuousAffineCoupling(
        latent_net=st.net.MLP(dim, [32], 2 * dim),
        time_net=st.net.TimeLinear(dim),
        mask='ordered_0',
        concatenate_time=False,
    ),
])

x = torch.randn(10, 4, dim)
t = torch.randn_like(x[...,:1])
y = f(x, t=t) # Outputs the same dimension as x
```

## Run tests

```
pytest --pyargs stribor
```
