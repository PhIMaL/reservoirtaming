from typing import Callable, Tuple
from flax import linen as nn
from flax.linen.initializers import normal, zeros
from .activation import leaky_erf
import jax.numpy as jnp


class RandomReservoir(nn.Module):
    """ Implements a generic reservoir."""

    n_reservoir: int
    input_scale: float = 0.4
    res_scale: float = 0.9
    bias_scale: float = 0.1

    activation_fn: Callable = leaky_erf
    activation_fn_args: Tuple = ()

    @nn.compact
    def __call__(self, state, x):
        # TODO: Turn state into flax variable.
        z_input = nn.Dense(
            self.n_reservoir,
            kernel_init=normal(self.input_scale),
            bias_init=normal(self.bias_scale),
        )
        z_res = nn.Dense(
            self.n_reservoir, kernel_init=normal(self.res_scale), use_bias=False
        )
        updated_state = self.activation_fn(
            z_input(x) + z_res(state), state, *self.activation_fn_args
        )

        return updated_state

    @staticmethod
    def initialize_state(rng, n_reservoir, init_fn=zeros):
        return init_fn(rng, (1, n_reservoir))

