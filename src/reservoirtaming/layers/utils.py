from flax import linen as nn
from jax import random, numpy as jnp
import numpy as np
from typing import Callable


class Diagonal(nn.Module):
    @nn.compact
    def __call__(self, X):
        D = self.param("kernel", random.rademacher, (1, X.shape[1]))
        return D * X


def hadamard(normalized=True, dtype=jnp.float32):
    """ We need the numpy to use it as initializer"""

    def init(key, shape, dtype=dtype):
        n = shape[0]
        # Input validation
        if n < 1:
            lg2 = 0
        else:
            lg2 = np.log2(n)
        assert 2 ** lg2 == n, "shape must be a positive integer and a power of 2."

        # Logic
        H = jnp.ones((1,), dtype=dtype)
        for i in np.arange(lg2):
            H = jnp.vstack([jnp.hstack([H, H]), jnp.hstack([H, -H])])

        if normalized:
            H = 2 ** (-lg2 / 2) * H
        return H

    return init


class HadamardTransform(nn.Module):
    n_hadamard: int

    @nn.compact
    def __call__(self, X):
        z = nn.Dense(self.n_hadamard, kernel_init=hadamard(), use_bias=False)(X)
        return z


class Log2Padding(nn.Module):
    padding_fn: Callable = jnp.zeros

    @nn.compact
    def __call__(self, X):
        n_in = X.shape[-1]
        next_power = int(2 ** jnp.ceil(jnp.log2(n_in)))  # finding next power of 2
        n_padding = int(next_power - n_in)
        return jnp.concatenate([X, self.padding_fn(*X.shape[:-1], n_padding)], axis=-1)
