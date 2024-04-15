import jax.numpy as jnp
from state_representations.base import StateRepresentationInterface


class TabularRepresentation(StateRepresentationInterface):
    def __init__(self, num_states, unit_norm=False):

        self.num_features = num_states

        self.features = jnp.eye(num_states)

        if unit_norm:
            self.features = jnp.divide(
                self.features, jnp.linalg.norm(self.features, axis=1).reshape((-1, 1))
            )

    def __getitem__(self, x):
        return self.features[x].squeeze()
