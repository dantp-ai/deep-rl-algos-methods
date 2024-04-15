import jax.numpy as jnp
from jax import random

from environments.base import BaseEnvironment

LEFT = 0
RIGHT = 1


class RandomWalkEnv(BaseEnvironment):
    """
    Random Walk from Sutton and Barto, 2018
    """

    def __init__(self):
        super().__init__()
        self.num_states = None

    def env_init(self, env_info={}):
        self.rand_generator = random.PRNGKey(env_info.get("seed"))
        self.num_states = env_info.get("num_states")
        self.log_episodes = env_info.get("log_episodes")

    def env_start(self):
        if self.log_episodes:
            self.experience_episode = []
        reward = 0
        self.num_timesteps = 0
        observation = jnp.array((self.num_states // 2,))
        is_terminal = False

        self.reward_obs_term = (reward, observation, is_terminal)

        return self.reward_obs_term[1]

    def env_step(self, action):
        last_state = self.reward_obs_term[1]
        if self.log_episodes:
            self.experience_episode.append(last_state)
        if action == LEFT:
            current_state = jnp.maximum(-1, last_state - 1)
        elif action == RIGHT:
            current_state = jnp.minimum(self.num_states, last_state + 1)
        else:
            raise Exception("Unexpected action given.")

        reward = 0
        is_terminal = False

        if jnp.array_equal(current_state, jnp.ones_like(current_state) * -1):
            reward = -1
            is_terminal = True
        elif jnp.array_equal(
            current_state, jnp.ones_like(current_state) * self.num_states
        ):
            reward = 1
            is_terminal = True

        self.reward_obs_term = (reward, current_state, is_terminal)

        return self.reward_obs_term

    def env_cleanup(self):
        pass

    def env_message(self, message):
        if message == "get episode" and self.log_episodes:
            return self.experience_episode
        raise Exception("Unexpected message given.")
