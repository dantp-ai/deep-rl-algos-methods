from abc import ABC, abstractmethod
import jax.numpy as jnp


class BasePolicy(ABC):
    @abstractmethod
    def get_action(self, observation):
        pass

