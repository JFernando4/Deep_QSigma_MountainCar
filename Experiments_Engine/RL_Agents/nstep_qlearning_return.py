import numpy as np
from Experiments_Engine.config import Config
from Experiments_Engine.Util import check_attribute_else_default


class nStep_QLearning_ReturnFunction:

    def __init__(self, config=None):

        assert isinstance(config, Config)
        """ 
        Parameters in config:
        Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
        n                       int             1                   the n of the n-step method
        gamma                   float           1.0                 the discount factor
        num_actions             int             3                   number of actions    
        """
        self.n = check_attribute_else_default(config, 'n', 1)
        self.gamma = check_attribute_else_default(config, 'gamma', 1.0)
        self.num_actions = check_attribute_else_default(config, 'num_actions', 3)

    def batch_iterative_return_function(self, rewards, actions, qvalues, terminations, timeouts, batch_size):
        """
        Assumptions of the implementation:
            All the rewards after the terminal state are 1.
            All the terminations indicators after the terminal state are True

        :param rewards: expected_shape = [batch_size, n]
        :param actions: expected_shape = [batch_size, n], expected_type = np.uint8, np.uint16, np.uint32, or np.uint64
        :param qvalues: expected_shape = [batch_size, n, num_actions]
        :param terminations: expected_shape = [batch_size, n]
        :param timeouts: expected_shape = [batch_size, n]
        :param batch_size: dtype = int
        :return: estimated_returns
        """
        num_actions = self.num_actions

        max_qvalues = np.max(qvalues, axis=2)  # The max qvalue at each time step
        batch_idxs = np.arange(batch_size)
        one_matrix = np.ones([batch_idxs.size, self.n], dtype=np.uint8)
        term_ind = terminations.astype(np.uint8)
        neg_term_ind = np.subtract(one_matrix, term_ind)
        timeout_ind = timeouts.astype(np.uint8)
        neg_timeout_ind = np.subtract(one_matrix, timeout_ind)
        estimated_Gt = neg_term_ind[:, -1] * neg_timeout_ind[:, -1] * max_qvalues[:, -1] + \
                       term_ind[:, -1] * neg_timeout_ind[:, -1] * rewards[:, -1] + \
                       neg_term_ind[:, -1] * timeout_ind[:, -1] * max_qvalues[:, -1]

        for i in range(self.n-1, -1, -1):
            R_t = rewards[:, i]
            A_t = actions[:, i]
            Q_t = qvalues[:, i, :]
            max_q = np.max(Q_t, axis=1)       # The maximum action-value
            assert np.sum(max_q == max_qvalues[:, i]) == batch_size

            G_t = R_t + self.gamma *  estimated_Gt
            timeout_factor = R_t + self.gamma * max_q
            # Since each transition in the buffer contains (R_{t+1}, S_{t+1}, A_{t+1}), and R_{t+1} = R(S_t, A_t),
            # this timeout factor is essentially computing:
            #  R_{t+1}
            #   + \gamma * \max_{a \in \mathcal{A}}  Q(S_{t+1}, a)
            # where \mathcal{A} is the set of all possible actions.

            estimated_Gt = neg_term_ind[:, i] * neg_timeout_ind[:, i] * G_t \
                           + term_ind[:, i] * neg_timeout_ind[:, i] * R_t \
                           + neg_term_ind[:, i] * timeout_ind[:, i] * timeout_factor

        return estimated_Gt
