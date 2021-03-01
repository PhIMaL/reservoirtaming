from flax.core import unfreeze
from jax.lax import scan


def train(model, state, params, train_data):
    def update_state(state, inputs):
        _, updated_state = model.apply(
            {"params": params, **state}, inputs, mutable=list(state.keys())
        )
        return updated_state, unfreeze(updated_state)["reservoir"]["state"]

    # Running forward pass
    state, reservoir_states = scan(update_state, state, train_data)
    return state, reservoir_states

