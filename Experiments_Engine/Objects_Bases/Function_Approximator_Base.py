import abc


class FunctionApproximatorBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        """ Initializes the environment """
        self._fa_dictionary = None
        return

    @abc.abstractmethod
    def update(self, state, action, nstep_return):
        """ Updates the function approximator """

    @abc.abstractmethod
    def get_value(self, state, action):
        """ Returns the approximation to the action-value or the state-value function """
        return 0.0

    @abc.abstractmethod
    def get_next_states_values(self, state):
        """ Returns all value functions of all the possible next states """
        return

    @abc.abstractmethod
    def get_fa_dictionary(self):
        return self._fa_dictionary
