from abc import ABC
from abc import abstractmethod

import jax.numpy as jnp


class BasePolicy(ABC):
    @abstractmethod
    def get_action(self, observation):
        pass


class RandomChainPolicy(BasePolicy):
    def get_action(self, observation):
        left = 0
        right = 1
        return jnp.random.choice([left, right])


class PWNorthEastPolicy(BasePolicy):
    def get_action(self, observation):
        north = 3
        east = 1
        return jnp.random.choice([north, east])


class MCFixedPolicy(BasePolicy):
    def get_action(self, observation):
        position, velocity = observation
        # https://github.com/openai/gym/wiki/MountainCar-v0
        # Add '+1' to convert to action values required by gym
        return int(jnp.sign(velocity) + 1)
