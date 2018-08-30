import numpy as np
from Experiments_Engine.config import Config
from Experiments_Engine.Util import check_attribute_else_default


class QSigmaReturnFunction:

    def __init__(self, tpolicy, config=None, bpolicy=None):

        assert isinstance(config, Config)
        """ 
        Parameters in config:
        Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
        n                       int             1                   the n of the n-step method
        gamma                   float           1.0                 the discount factor
        compute_bprobs        bool            False                 whether to recompute behaviour policy probabilities 
                                                                    or use the ones stored in the buffer.
        onpolicy                bool            True                whether to compute the on-policy return or the
                                                                    off-policy, i.e. compute the importance sampling
                                                                    ratio or not.
        """
        self.n = check_attribute_else_default(config, 'n', 1)
        self.gamma = check_attribute_else_default(config, 'gamma', 1.0)
        self.compute_bprobs = check_attribute_else_default(config, 'compute_bprobs', False)
        self.onpolicy = check_attribute_else_default(config, 'onpolicy', True)

        """
        Other Parameters:
        tpolicy - The target policy
        bpolicy - Behaviour policy. Only required if compute_bprobs is True.
        """
        self.tpolicy = tpolicy
        self.bpolicy = bpolicy
        if self.compute_bprobs:
            assert self.bpolicy is not None

    def batch_iterative_return_function(self, rewards, actions, qvalues, terminations, bprobs, sigmas,
                                        batch_size):
        """
        Assumptions of the implementation:
            All the rewards after the terminal state are 1.
            All the terminations indicators after the terminal state are True
            All the bprobabilities and tprobabilities after the terminal state are 1

        :param rewards: expected_shape = [batch_size, n]
        :param actions: expected_shape = [batch_size, n], expected_type = np.uint8, np.uint16, np.uint32, or np.uint64
        :param qvalues: expected_shape = [batch_size, n, num_actions]
        :param terminations: expected_shape = [batch_size, n]
        :param bprobs: expected_shape = [batch_size, n, num_actions]
        :param sigmas: expected_shape = [batch_size, n]
        :param batch_size: dtype = int
        :return: estimated_returns
        """
        num_actions = self.tpolicy.num_actions
        tprobabilities = np.ones([batch_size, self.n, self.tpolicy.num_actions], dtype=np.float64)
        if not self.onpolicy:
            bprobs = bprobs if not self.compute_bprobs \
                         else np.ones([batch_size, self.n, self.tpolicy.num_actions], dtype=np.float64)

        for i in range(self.n):
            tprobabilities[:, i] = self.tpolicy.batch_probability_of_action(qvalues[:,i])
            if self.compute_bprobs:
                bprobs[:, i] = self.bpolicy.batch_probability_of_action(qvalues[:,i])

        selected_qval = qvalues.take(np.arange(actions.size) * num_actions + actions.flatten()).reshape(actions.shape)
        batch_idxs = np.arange(batch_size)
        one_vector = np.ones(batch_idxs.size)
        one_matrix = np.ones([batch_idxs.size, self.n], dtype=np.uint8)
        term_ind = terminations.astype(np.uint8)
        neg_term_ind = np.subtract(one_matrix, term_ind)
        estimated_Gt = neg_term_ind[:, -1] * selected_qval[:, -1] + term_ind[:, -1] * rewards[:, -1]

        for i in range(self.n-1, -1, -1):
            R_t = rewards[:, i]
            A_t = actions[:, i]
            Q_t = qvalues[:, i, :]
            Sigma_t = sigmas[:, i]
            exec_q = Q_t[batch_idxs, A_t]       # The action-value of the executed actions
            assert np.sum(exec_q == selected_qval[:, i]) == batch_size
            tprob = tprobabilities[:, i, :]     # The probability of the executed actions under the target policy
            exec_tprob = tprob[batch_idxs, A_t]
            if not self.onpolicy:
                bprob = bprobs[:, i, :]
                exec_bprob = bprob[batch_idxs, A_t] # The probability of the executed actions under the behaviour policy
                rho = np.divide(exec_tprob, exec_bprob)
            else:
                rho = 1.0
            V_t = np.sum(np.multiply(Q_t, tprob), axis=-1)

            G_t = R_t + self.gamma * (rho * Sigma_t + (one_vector - Sigma_t) * exec_tprob) * estimated_Gt +\
                  self.gamma * (one_vector - Sigma_t) * (V_t - exec_tprob * exec_q)
            estimated_Gt = neg_term_ind[:, i] * G_t + term_ind[:, i] * R_t

        return estimated_Gt
