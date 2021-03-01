from flax import linen as nn
from flax.linen.initializers import normal


class RandomReservoir(nn.Module):
    """ Implements a generic reservoir."""

    n_res: int
    input_scale: float = 0.4
    res_scale: float = 0.9
    bias_scale: float = 0.1

    @nn.compact
    def __call__(self, inputs, state):
        z_input = nn.Dense(
            self.n_res,
            kernel_init=normal(self.input_scale),
            bias_init=normal(self.bias_scale),
        )
        z_res = nn.Dense(self.n_res, kernel_init=normal(self.res_scale), use_bias=False)

        return z_input(inputs) + z_res(state)
