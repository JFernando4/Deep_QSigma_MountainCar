from numpy import inf, zeros
import numpy as np

from Experiments_Engine.Objects_Bases import RL_ALgorithmBase
from Experiments_Engine.Util import check_attribute_else_default, check_dict_else_default
from Experiments_Engine.config import Config


class QSigma(RL_ALgorithmBase):

    def __init__(self, environment, function_approximator, target_policy, behaviour_policy, config=None, er_buffer=None,
                 summary=None):
        super().__init__()
        """
        Summary Name: return_per_episode
        """
        self.config = config or Config()
        assert isinstance(config, Config)
        """ 
        Parameters in config:
        Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
        n                       int             1                   the n of the n-step method
        gamma                   float           1.0                 the discount factor
        beta                    float           1.0                 the decay factor of sigma
        sigma                   float           0.5                 see De Asis et.al. in AAAI 2018 proceedings
        use_er_buffer           bool            False               indicates whether to use experience replay buffer
        initial_rand_steps      int             0                   number of random steps before training starts
        rand_steps_count        int             0                   number of random steps taken so far
        save_summary            bool            False               Save the summary of the agent (return per episode)
        """
        self.n = check_attribute_else_default(self.config, 'n', 1)
        self.gamma = check_attribute_else_default(self.config, 'gamma', 1.0)
        self.beta = check_attribute_else_default(self.config, 'beta', 1.0)
        self.sigma = check_attribute_else_default(self.config, 'sigma', 0.5)
        self.use_er_buffer = check_attribute_else_default(self.config, 'use_er_buffer', False)
        self.initial_rand_steps = check_attribute_else_default(self.config, 'initial_rand_steps', 0)
        check_attribute_else_default(self.config, 'rand_steps_count', 0)
        self.save_summary = check_attribute_else_default(self.config, 'save_summary', False)

        if self.save_summary:
            assert isinstance(summary, dict)
            self.summary = summary
            check_dict_else_default(self.summary, 'return_per_episode', [])

        " Other Parameters "
        # Behaviour and Target Policies
        self.bpolicy = behaviour_policy
        self.tpolicy = target_policy

        # Experience Replay Buffer: used for storing and retrieving observations. Mainly for Deep RL
        self.er_buffer = er_buffer
        if self.use_er_buffer: assert self.er_buffer is not None

        # Function Approximator: used to approximate the Q-Values
        self.fa = function_approximator

        # Environment that the agent is interacting with
        self.env = environment

    def iterative_return_function(self, trajectory):
        trajectory_len = len(trajectory)
        last_reward, last_action, last_qvalues, last_termination = trajectory[trajectory_len-1]
        estimate_return = last_reward if last_termination else last_qvalues[last_action]
        for i in range(trajectory_len-1, -1, -1):
            R_t, A_t, Q_t, termination = trajectory[i]
            if termination:
                estimate_return = R_t
            else:
                Sigma_t = self.sigma
                tprobs = self.tpolicy.probability_of_action(Q_t, all_actions=True)
                bprobs = self.bpolicy.probability_of_action(Q_t, all_actions=True)
                assert bprobs[A_t] != 0
                rho = tprobs[A_t]/bprobs[A_t]
                v_t = np.sum(np.multiply(tprobs, Q_t))
                pi_t = tprobs[A_t]
                estimate_return = R_t + self.gamma * (rho * Sigma_t + (1-Sigma_t) * pi_t) * estimate_return +\
                                  self.gamma * (1-Sigma_t) * (v_t - pi_t * Q_t[A_t])
        return estimate_return

    def adjust_sigma(self):
        self.config.sigma *= self.beta
        self.sigma = self.config.sigma

    def train(self, num_episodes):
        if num_episodes == 0: return

        Actions = zeros(self.n + 1, dtype=int)
        States = [[] for _ in range(self.n + 1)]
        for episode in range(num_episodes):
            # Record Keeping
            episode_reward_sum = 0
            # Current State, Action, and Q_values
            S = self.env.get_current_state()
            q_values = self.fa.get_next_states_values(S)
            if self.config.rand_steps_count >= self.initial_rand_steps:
                A = self.bpolicy.choose_action(q_values)
                self.tpolicy.anneal()
                self.bpolicy.anneal()
            else:
                A = np.random.randint(len(q_values))
                self.config.rand_steps_count += 1
            # Storing in the experience replay buffer
            if self.use_er_buffer:
                observation = {"reward": 0, "action":A, "state":self.env.get_state_for_er_buffer(), "terminate": False,
                               "bprobabilities": np.zeros(q_values.shape), "sigma":self.sigma}
                self.er_buffer.store_observation(observation)
            T = inf
            t = 0
            # Storing
            States[t % (self.n + 1)] = S
            Actions[t % (self.n + 1)] = A
            # Trajectory
            trajectory = []
            while 1:
                if t < T:
                    # Step in the environment
                    S, R, terminate = self.env.update(A)
                    # Updating Q_values and State
                    States[(t+1) % (self.n+1)] = S
                    q_values = self.fa.get_next_states_values(S)
                    # Record Keeping
                    episode_reward_sum += R
                    if terminate:
                        T = t + 1
                        bpropabilities = np.ones(self.env.get_num_actions(), dtype=np.float64)
                        A = np.uint8(0)
                    else:
                        if self.config.rand_steps_count >= self.initial_rand_steps:
                            A = self.bpolicy.choose_action(q_values)
                            bpropabilities = self.bpolicy.probability_of_action(q_values, all_actions=True)
                            self.tpolicy.anneal()
                            self.bpolicy.anneal()
                        else:
                            A = np.random.randint(len(q_values))
                            bpropabilities = np.ones(self.env.get_num_actions(), dtype=np.float64) * (1/self.env.get_num_actions())
                            self.config.rand_steps_count += 1

                        Actions[(t + 1) % (self.n + 1)] = A
                    # Storing Trajectory
                    trajectory.append([R, A, q_values, terminate])
                    # Storing in the experience replay buffer
                    if self.use_er_buffer:
                        observation = {"reward": R, "action": A, "state": self.env.get_state_for_er_buffer(),
                                       "terminate": terminate, "bprobabilities": bpropabilities, "sigma": self.sigma}
                        self.er_buffer.store_observation(observation)
                # Computing the return and updating the function approximator
                tau = t - self.n + 1
                if tau >= 0:
                    if len(trajectory) >= 1:
                        if not self.use_er_buffer:
                            G = self.iterative_return_function(trajectory)
                        else:   # No need to compute the return if we're using experience replay
                            G = 0
                        if self.config.rand_steps_count >= self.initial_rand_steps:
                            self.fa.update(States[tau % (self.n+1)], Actions[tau % (self.n+1)], nstep_return=G)
                        trajectory.pop(0)
                t += 1
                if tau == T-1: break
            # End of episode
            if self.save_summary: self.summary['return_per_episode'].append(episode_reward_sum)
            if self.config.rand_steps_count >= self.initial_rand_steps: self.adjust_sigma()
            self.env.reset()
