from typing import Callable, Tuple
from flax import linen as nn
from flax.linen.initializers import normal, zeros
from .activation import leaky_erf
import jax.numpy as jnp
from .utils import Diagonal, HadamardTransform


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


class StructuredTransform(nn.Module):
    n_reservoir: int
    n_input: int
    n_layers: int = 3

    input_scale: float = 0.4
    res_scale: float = 0.9
    bias_scale: float = 0.1

    activation_fn: Callable = leaky_erf
    activation_fn_args: Tuple = (1.0,)

    def setup(self):
        # Padding
        self.n_hadamard = int(
            2 ** jnp.ceil(jnp.log2(self.n_input + self.n_reservoir))
        )  # finding next power of 2
        self.n_padding = int(self.n_hadamard - self.n_reservoir - self.n_input)
        self.padding = jnp.zeros((1, self.n_padding))

        # Layers
        self.diagonal_layers = [Diagonal() for _ in jnp.arange(self.n_layers)]
        self.hadamard = HadamardTransform(self.n_hadamard)
        self.bias = self.param(
            "bias", normal(stddev=self.bias_scale), (self.n_reservoir,)
        )

    def __call__(self, state, inputs):
        X = jnp.concatenate(
            [self.res_scale * state, self.input_scale * inputs, self.padding], axis=1
        )
        for diagonal in self.diagonal_layers:
            X = self.hadamard(diagonal(X))

        # TODO: check if self.n_hadamard is correct; comes from code from paper
        z = X[:, : self.n_reservoir] / self.n_hadamard + self.bias
        z = self.activation_fn(z, state, *self.activation_fn_args)
        return z

    @staticmethod
    def initialize_state(rng, n_reservoir, init_fn=zeros):
        return init_fn(rng, (1, n_reservoir))
