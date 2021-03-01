from flax import linen as nn
import jax.numpy as jnp
from typing import Callable, Tuple


class GenericEchoState(nn.Module):
    reservoir: nn.Module
    act_fn: Callable
    n_reservoir: int
    reservoir_args: Tuple
    act_fn_args: Tuple

    @nn.compact
    def __call__(self, inputs):
        # Initializing internal reservoir state
        is_initialized = self.has_variable("reservoir", "state")
        reservoir_state = self.variable(
            "reservoir", "state", lambda n: jnp.zeros((n,)), self.n_reservoir
        )

        # Calculating new state
        state = reservoir_state.value
        z = self.reservoir(self.n_reservoir, *self.reservoir_args)(inputs, state)
        updated_state = self.act_fn(z, state, *self.act_fn_args)

        # Updating internal state
        if is_initialized:
            reservoir_state.value = updated_state

        return updated_state

