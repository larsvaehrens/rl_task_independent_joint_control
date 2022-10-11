from abc import ABC, abstractmethod


class VecEnv(ABC):

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations.

        :return: ([numpy.ndarray]) observation (num_envs x obs_dim)
        """
        pass

    @abstractmethod
    def step(self, actions):
        """
        Step the environments with the given action

        :param actions: ([numpy.ndarray]) the actions (num_envs x actions_dim)
        :return: ([numpy.ndarray], [numpy.ndarray], [numpy.ndarray], dict) observation, reward, done, information
        """
        pass

    @property
    def unwrapped(self):
        return self
