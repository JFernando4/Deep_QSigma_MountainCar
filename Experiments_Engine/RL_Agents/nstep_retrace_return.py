import numpy as np
from Experiments_Engine.config import Config
from Experiments_Engine.Util import check_attribute_else_default


class nStep_Retrace_ReturnFunction:

    def __init__(self, tpolicy, bpolicy, config=None):

        assert isinstance(config, Config)
        """ 
        Parameters in config:
        Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
        n                       int             1                   the n of the n-step method
        gamma                   float           1.0                 the discount factor
        """
        self.n = check_attribute_else_default(config, 'n', 1)
        self.gamma = check_attribute_else_default(config, 'gamma', 1.0)

        """
        Other Parameters:
        tpolicy - The target policy
        bpolicy - Behaviour policy. Only required if compute_bprobs is True.
        """
        self.tpolicy = tpolicy
        self.bpolicy = bpolicy

    def batch_iterative_return_function(self, rewards, actions, qvalues, terminations, timeouts, bprobs, batch_size):
        """
        Assumptions of the implementation:
            All the rewards after the terminal state are 1.
            All the terminations indicators after the terminal state are True
            All the bprobabilities and tprobabilities after the terminal state are 1

        :param rewards: expected_shape = [batch_size, n]
        :param actions: expected_shape = [batch_size, n], expected_type = np.uint8, np.uint16, np.uint32, or np.uint64
        :param qvalues: expected_shape = [batch_size, n, num_actions]
        :param terminations: expected_shape = [batch_size, n]
        :param timeouts: expected_shape = [batch_size, n]
        :param bprobs: expected_shape = [batch_size, n, num_actions]
        :param batch_size: dtype = int
        :return: estimated_returns
        """
        num_actions = self.tpolicy.num_actions
        tprobabilities = np.ones([batch_size, self.n, self.tpolicy.num_actions], dtype=np.float64)

        for i in range(self.n):
            tprobabilities[:, i] = self.tpolicy.batch_probability_of_action(qvalues[:,i])

        selected_qval = qvalues.take(np.arange(actions.size) * num_actions + actions.flatten()).reshape(actions.shape)
        batch_idxs = np.arange(batch_size)
        one_vector = np.ones(batch_idxs.size)
        one_matrix = np.ones([batch_idxs.size, self.n], dtype=np.uint8)
        term_ind = terminations.astype(np.uint8)
        neg_term_ind = np.subtract(one_matrix, term_ind)
        timeout_ind = timeouts.astype(np.uint8)
        neg_timeout_ind = np.subtract(one_matrix, timeout_ind)
        estimated_Gt = neg_term_ind[:, -1] * neg_timeout_ind[:, -1] * selected_qval[:, -1] + \
                       term_ind[:, -1] * neg_timeout_ind[:, -1] * rewards[:, -1] + \
                       neg_term_ind[:, -1] * timeout_ind[:, -1] * selected_qval[:, -1]

        for i in range(self.n-1, -1, -1):
            R_t = rewards[:, i]
            A_t = actions[:, i]
            Q_t = qvalues[:, i, :]
            exec_q = Q_t[batch_idxs, A_t]       # The action-value of the executed actions
            assert np.sum(exec_q == selected_qval[:, i]) == batch_size
            tprob = tprobabilities[:, i, :]     # The probability of the executed actions under the target policy
            exec_tprob = tprob[batch_idxs, A_t]
            bprob = bprobs[:, i, :]
            exec_bprob = bprob[batch_idxs, A_t] # The probability of the executed actions under the behaviour policy
            rho = np.divide(exec_tprob, exec_bprob)
            truncated_rho = np.min(np.column_stack((rho, one_vector)), axis=1)

            V_t = np.sum(np.multiply(Q_t, tprob), axis=-1)
            G_t = R_t + self.gamma * (truncated_rho * (estimated_Gt - exec_q) + V_t)
            timeout_factor = R_t + self.gamma * V_t
            # Since each transition in the buffer contains (R_{t+1}, S_{t+1}, A_{t+1}), and R_{t+1} = R(S_t, A_t),
            # this timeout factor is essentially computing:
            #  R_{t+1}
            #   + \gamma * \sum_{a \in \mathcal{A}} \pi(a|S_{t+1}) Q(S_{t+1}, a),
            # where \mathcal{A} is the set of all possible actions.
            estimated_Gt = neg_term_ind[:, i] * neg_timeout_ind[:, i] * G_t \
                           + term_ind[:, i] * neg_timeout_ind[:, i] * R_t \
                           + neg_term_ind[:, i] * timeout_ind[:, i] * timeout_factor

        return estimated_Gt
