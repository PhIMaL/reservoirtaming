from jax import numpy as jnp, jit
from jax.scipy.special import erf
from functools import partial


@partial(jit, static_argnums=(0,))
def generic_leaky(f, leak_rate, z, state):
    n_reservoir = z.shape[0]
    updated_state = (1.0 - leak_rate) * state + leak_rate * f(z) / jnp.sqrt(n_reservoir)
    return updated_state


@jit
def leaky_erf(z, state, leak_rate):
    return generic_leaky(erf, leak_rate, z, state)
