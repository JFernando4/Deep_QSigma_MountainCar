from numpy.random import uniform, randint
from numpy import array, zeros
import numpy as np

from Experiments_Engine.Objects_Bases import PolicyBase
from Experiments_Engine.config import Config
from Experiments_Engine.Util import check_attribute_else_default

class EpsilonGreedyPolicy(PolicyBase):

    def __init__(self, config=None, behaviour_policy=False):
        super().__init__()
        """ 
        Parameters in config:
        Name:               Type:           Default:            Description: (Omitted when self-explanatory)
        num_actions         int             2                   Number of actions available to the agent
        initial_epsilon     float           0.1                 Epsilon before annealing
        anneal_epsilon      bool            False               Indicates whether to anneal epsilon
        final_epsilon       float           initial_epsilon     The value of epsilon after annealing    
        annealing_period    int             100,000             Number of steps before reaching final epsilon
        anneal_steps_count  int             0                   Number of times epsilon has been annealed  
                
        Other Parameters:
        Name:               Type:           Default:            Description:
        behaviour_policy    bool            False               Indicates whether this is the behaviour or target policy
        """
        self.config = config or Config()
        assert isinstance(config, Config)
        self.num_actions = check_attribute_else_default(self.config, 'num_actions', 2)
        if behaviour_policy:
            current_config = check_attribute_else_default(self.config, 'behaviour_policy', Config())
        else:
            current_config = check_attribute_else_default(self.config, 'target_policy', Config())
        self.initial_epsilon = check_attribute_else_default(current_config, 'initial_epsilon', 0.1)
        self.anneal_epsilon = check_attribute_else_default(current_config, 'anneal_epsilon', False)
        self.final_epsilon = check_attribute_else_default(current_config, 'final_epsilon', self.initial_epsilon)
        self.annealing_period = check_attribute_else_default(current_config, 'annealing_period', 100000)
        check_attribute_else_default(self.config, 'anneal_steps_count', 0)

        self.epsilon = self.initial_epsilon
        self.p_random = (self.epsilon / self.num_actions)
        self.p_optimal = self.p_random + (1 - self.epsilon)

    """ Chooses an action from q according to the probability epsilon"""
    def choose_action(self, q_value):
        p = uniform()
        if True in (np.array(q_value) == np.inf):
            raise ValueError("One of the Q-Values has a value of infinity.")
        if p < self.epsilon:
            action = randint(self.num_actions, dtype=np.uint8)
        else:
            # choosing a random action from all the possible maximum action
            action = np.uint8(np.random.choice(np.argwhere(q_value == np.max(q_value)).flatten(), size=1)[0])
        return action

    """" Returns the probability of a given action or of all the actions """
    def probability_of_action(self, q_values, action=0, all_actions=False):
        assert isinstance(q_values, np.ndarray)
        max_q = np.max(q_values)
        total_max_actions = np.sum(max_q == array(q_values))
        action_probabilities = zeros(self.num_actions, dtype=np.float64) + self.p_random

        """ Sanity Check:
        Let p_random = epsilon / (#actions), p_optimal = p_random + (1 - epsilon), and (#optimal) and (#actions) 
        be the total number of optimal actions and total number of actions, respectively. Then we have:
            
        \sum_{a != optimal} p_random + \sum_{a == optimal} [(p_optimal + (#optimal -1) p_random] / (#optimal)   =
            p_random * (#actions - #optimal) + (#optimal) [p_optimal + (#optimal-1) p_random] / (#optimal)      =
            (#actions) p_random) + p_optimal - p_random = epsilon + (1-epsilon) + p_random - p_random = 1
            
        as long as epsilon \in [0,1]
        """
        action_probabilities[np.squeeze(np.argwhere(q_values == max_q))] = \
            (self.p_optimal + (total_max_actions - 1) * self.p_random) / total_max_actions

        if all_actions:
            return action_probabilities
        else:
            return action_probabilities[action]

    def batch_probability_of_action(self, q_values):
        max_qs = np.max(q_values, axis=1)
        equal_to_max_qs = np.equal(q_values, max_qs[:, None])
        total_max_actions = np.sum(equal_to_max_qs, axis=1)
        max_actions_indices = np.argwhere(equal_to_max_qs)

        action_probabilities = np.zeros(q_values.shape, dtype=np.float64) + self.p_random
        action_probabilities[max_actions_indices[:, 0], max_actions_indices[:,1]] = \
            np.divide(self.p_optimal + (total_max_actions[max_actions_indices[:, 0]] - 1) * self.p_random,
                      total_max_actions[max_actions_indices[:, 0]])
        return action_probabilities

    """ Moves one step closer to the final epsilon """
    def anneal(self):
        if self.anneal_epsilon:
            if self.config.anneal_steps_count < self.annealing_period:
                self.epsilon = self.initial_epsilon - ((self.initial_epsilon - self.final_epsilon) *
                               min(1, self.config.anneal_steps_count / self.annealing_period))
                self.config.anneal_steps_count += 1
            else:
                self.epsilon = self.final_epsilon
            self.p_random = (self.epsilon / self.num_actions)
            self.p_optimal = self.p_random + (1 - self.epsilon)
            return
        else: return
