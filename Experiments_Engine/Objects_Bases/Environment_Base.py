import abc


class EnvironmentBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        """ Initializes the environment """
        return

    @abc.abstractmethod
    def reset(self):
        """ Resets the environment and returns the initial state """
        return

    @abc.abstractmethod
    def update(self, A):
        """"
        Given an action, it returns the new state and the corresponding reward.
        It must also return a flag indicating whether the terminal state has been reached.
        """
        return

    @abc.abstractmethod
    def get_num_actions(self):
        """ Returns the number of actions available to the agent """
        return

    @abc.abstractmethod
    def get_current_state(self):
        """ Returns the current state of the environment """
        return

    @abc.abstractmethod
    def get_actions(self):
        """ Returns the actions available in the environment """
        return

    @abc.abstractmethod
    def set_render(self, *args):
        """ Set render to true if the environment has the option to render """
        return