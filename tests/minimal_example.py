from reservoirtaming.layers.reservoirs import RandomReservoir
from reservoirtaming.models.generic import EchoState
from reservoirtaming.data.KS import KS
from jax import random
import numpy as np


# Setting up our dataset; similar to jonathans
L = 22 / (2 * np.pi)  # length
N = 100  # space discretization step
dt = 0.25  # time discretization step
N_train = 10000
N_test = 2000
N_init = 1000  # remove the initial points
tend = (N_train + N_test) * dt + N_init

np.random.seed(1)
dns = KS(L=L, N=N, dt=dt, tend=tend)
dns.simulate()

# Prepping train and test matrices
_, u_train, u_test, _ = np.split(
    dns.uu / np.sqrt(N),
    [N_init, (N_init + N_train), (N_init + N_train + N_test)],
    axis=0,
)

u_train = np.expand_dims(u_train, 1)
print(u_train.shape)
key = random.PRNGKey(42)
model = EchoState(4000, 100, RandomReservoir)
state = model.initialize_state(key, 4000)
params = model.init(key, state, u_train[0])
