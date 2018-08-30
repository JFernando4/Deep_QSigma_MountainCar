import abc


class RL_ALgorithmBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        """ Initializes the agent """
        pass

    @abc.abstractmethod
    def update(self, reward):
        """ Updates the algorithm based on the reward and the state """
        return

    @abc.abstractmethod
    def get_return_per_episode(self):
        """ Returns the cummulative reward so far """
        return

    @abc.abstractmethod
    def get_average_reward(self):
        """ Returns the average reward so far """
        return

    @abc.abstractmethod
    def reset_episode_history(self):
        """ Resets the episode history """
        return

