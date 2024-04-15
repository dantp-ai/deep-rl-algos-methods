from abc import ABCMeta
from abc import abstractmethod


class BaseEnvironment(metaclass=ABCMeta):
    """Abstract environment base class for RL-Glue-py."""

    @abstractmethod
    def __init__(self):
        self.rand_generator = None
        reward = None
        observation = None
        termination = None
        self.reward_obs_term = (reward, observation, termination)

    @abstractmethod
    def env_init(self, env_info={}):
        """Setup for the environment called when the experiment first starts."""

    @abstractmethod
    def env_start(self):
        """The first method called when the experiment starts, called before the agent starts."""

    @abstractmethod
    def env_step(self, action):
        """A step taken by the environment."""

    @abstractmethod
    def env_cleanup(self):
        """Cleanup done after the environment ends"""

    @abstractmethod
    def env_message(self, message):
        """A message asking the environment for information"""
