import abc


class PolicyBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        """ Initializes the policy """
        pass

    @abc.abstractmethod
    def choose_action(self, q_value):
        """ Chooses an action from a list of action values """
        return

    @abc.abstractmethod
    def update_policy(self):
        """ Updates the policy """
        return

    @abc.abstractmethod
    def probability_of_action(self, q_values, all_actions, action=0):
        """ Given q_value it returns the probability of each action if all_actions is true, otherwise
            it returns the probability of action """
        return [0.0 for _ in q_values]
