import abc
from Experiments_Engine.Objects_Bases import EnvironmentBase
from Experiments_Engine.Objects_Bases import FunctionApproximatorBase

class AgentBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, env=EnvironmentBase(), fa=FunctionApproximatorBase()):
        """ Initializes the agent """
        self.env = env
        self.fa = fa
        pass

    @abc.abstractmethod
    def step(self):
        """ Moves one step forward in time """
        pass

    @abc.abstractmethod
    def run_episode(selfs):
        """ Runs a full episode beginning to end """
        return

    @abc.abstractmethod
    def train(self, number_of_episodes):
        """ Runs (number_of_episodes) episodes beginning to end """
        return

    @abc.abstractmethod
    def get_agent_dictionary(self):
        """ Returns a dictionary with all the hyperparameters of the agent """
        return
