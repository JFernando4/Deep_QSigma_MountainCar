import numpy as np

from Experiments_Engine.Function_Approximators.Neural_Networks.NN_Utilities import CircularBuffer
from Experiments_Engine.RL_Agents.nstep_retrace_return import nStep_Retrace_ReturnFunction
from Experiments_Engine.Util.utils import check_attribute_else_default
from Experiments_Engine.config import Config


class nStep_Retrace_ExperienceReplay:

    def __init__(self, config, return_function):

        """ Parameters:
        Name:               Type:           Default:            Description: (Omitted when self-explanatory)
        buff_sz             int             10                  buffer size
        batch_sz            int             1
        env_state_dims      list            [2,2]               dimensions of the observations to be stored in the buffer
        num_actions         int             2                   number of actions available to the agent
        obs_dtype           np.type         np.uint8            the data type of the observations
        initial_rand_steps  int             0                   number of random steps before decaying sigma
        rand_steps_count    int             0                   number of random steps taken so far
        store_return        bool            True                save the computed return so that it can be reused
        """
        assert isinstance(config, Config)
        self.config = config
        self.buff_sz = check_attribute_else_default(self.config, 'buff_sz', 10)
        self.batch_sz = check_attribute_else_default(self.config, 'batch_sz', 1)
        self.env_state_dims = list(check_attribute_else_default(self.config, 'env_state_dims', [2,2]))
        self.num_actions = check_attribute_else_default(self.config, 'num_actions', 2)
        self.obs_dtype = check_attribute_else_default(self.config, 'obs_dtype', np.uint8)
        self.initial_rand_steps = check_attribute_else_default(self.config, 'initial_rand_steps', 0)
        check_attribute_else_default(self.config, 'rand_steps_count', 0)
        self.store_return = check_attribute_else_default(self.config, 'store_return', True)

        """ Parameters for Return Function """
        assert isinstance(return_function, nStep_Retrace_ReturnFunction)
        self.return_function = return_function
        self.n = return_function.n

        """ Termination or Timeout Count for Applying the Decay on Sigma """
        self.episodes_since_last_decay = 0

        """ Parameters to keep track of the current state of the buffer """
        self.current_index = 0
        self.full_buffer = False

        """ Circular Buffers """
        self.state = CircularBuffer(self.buff_sz, shape=tuple(self.env_state_dims), dtype=self.obs_dtype)
        self.action = CircularBuffer(self.buff_sz, shape=(), dtype=np.uint8)
        self.reward = CircularBuffer(self.buff_sz, shape=(), dtype=np.int32)
        self.terminate = CircularBuffer(self.buff_sz, shape=(), dtype=np.bool)
        self.timeout = CircularBuffer(self.buff_sz, shape=(), dtype=np.bool)
        self.bprobabilities = CircularBuffer(self.buff_sz, shape=(self.num_actions,), dtype=np.float64)
        self.estimated_return = CircularBuffer(self.buff_sz, shape=(), dtype=np.float64)
        self.up_to_date = CircularBuffer(self.buff_sz, shape=(), dtype=np.bool)

    def store_observation(self, observation):
        """ The only two keys that are required are 'state' """
        assert isinstance(observation, dict)
        assert all(akey in observation.keys() for akey in ["reward", "action", "state", "terminate"])

        temp_terminate = observation['terminate']
        temp_timeout = observation['timeout']
        reward = observation["reward"]

        self.state.append(observation["state"])
        self.action.append(observation["action"])
        self.reward.append(reward)
        self.terminate.append(temp_terminate)
        self.timeout.append(temp_timeout)
        assert hasattr(self, 'bprobabilities')
        assert 'bprobabilities' in observation.keys()
        self.bprobabilities.append(observation["bprobabilities"])
        self.estimated_return.append(0.0)
        self.up_to_date.append(False)

        self.current_index += 1
        if self.current_index >= self.buff_sz:
            self.current_index = 0
            self.full_buffer = True

    def sample_indices(self):
        bf_start = self.terminate.start
        inds_start = 0
        if not self.full_buffer:
            inds_end = self.current_index - (self.n+1)
        else:
            inds_end = self.buff_sz - 1 - (self.n + 1)
        sample_inds = np.random.randint(inds_start, inds_end, size=self.batch_sz)
        terminations_timeout = np.logical_or(self.terminate.data.take(bf_start + sample_inds, axis=0, mode='wrap'),
                                             self.timeout.data.take(bf_start + sample_inds, axis=0, mode='wrap'))
        terminations_timeout_sum = np.sum(terminations_timeout)
        while terminations_timeout_sum != 0:
            bad_inds = np.squeeze(np.argwhere(terminations_timeout))
            new_inds = np.random.randint(inds_start, inds_end, size=terminations_timeout_sum)
            sample_inds[bad_inds] = new_inds
            terminations_timeout = np.logical_or(self.terminate.data.take(bf_start + sample_inds, axis=0, mode='wrap'),
                                                 self.timeout.data.take(bf_start + sample_inds, axis=0, mode='wrap'))
            terminations_timeout_sum = np.sum(terminations_timeout)
        return sample_inds

    def get_data(self, update_function):
        indices = self.sample_indices()
        bf_start = self.action.start

        estimated_returns = np.zeros(self.batch_sz, dtype=np.float64)

        sample_states = np.zeros((self.batch_sz, 1) + tuple(self.env_state_dims), dtype=self.obs_dtype)
        sample_actions = self.action.data.take(bf_start + indices, mode='wrap', axis=0)
        # Abbreviations: tj = trajectory, tjs = trajectories
        tjs_states = np.zeros(shape=(self.batch_sz * self.n, 1) + tuple(self.env_state_dims),
                              dtype=self.obs_dtype)
        tjs_actions = np.zeros(self.batch_sz * self.n, np.uint8)
        tjs_rewards = np.zeros(self.batch_sz * self.n, np.int32)
        tjs_terminations = np.ones(self.batch_sz * self.n, np.bool)
        tjs_timeout = np.zeros(self.batch_sz * self.n, np.bool)
        tjs_bprobabilities = np.ones([self.batch_sz * self.n, self.num_actions], np.float64)

        batch_idx = 0
        tj_start_idx = 0
        retrieved_count = 0
        computed_return_buffer_inds = np.zeros(self.batch_sz, dtype=np.int64)
        computed_return_batch_inds = np.zeros(self.batch_sz, dtype=np.int64)
        for idx in indices:
            assert not self.terminate[idx] and not self.timeout[idx]
            start_idx = idx
            # First terminal state from the left. Reversed because we want to find the first terminal state before
            # the current state
            left_terminal_idx = 0

            if self.up_to_date.data.take(bf_start + idx, axis=0, mode='wrap') and self.store_return:
                estimated_returns[batch_idx] = self.estimated_return[idx]
                sample_state = self.state.data.take(bf_start + start_idx + np.arange(1), mode='wrap',
                                                    axis=0)
                sample_states[batch_idx] += sample_state
                retrieved_count += 1
                batch_idx += 1
            else:
                # First terminal or timeout state from center to right
                right_terminal = np.logical_or(
                    self.terminate.data.take(bf_start + idx + np.arange(self.n + 1), mode='wrap', axis=0),
                    self.timeout.data.take(bf_start + idx + np.arange(self.n + 1), mode='wrap', axis=0))
                right_terminal_true_idx = np.argmax(right_terminal)
                right_terminal_stop = self.n if right_terminal_true_idx == 0 else right_terminal_true_idx

                # trajectory indices
                tj_end_idx = tj_start_idx + right_terminal_stop - 1
                tj_slice = slice(tj_start_idx, tj_end_idx + 1)
                tj_indices = idx + 1 + np.arange(right_terminal_stop)

                # Collecting: trajectory actions, rewards, terminations, bprobabilities, and sigmas
                tjs_actions[tj_slice] = self.action.data.take(bf_start + tj_indices, axis=0, mode='wrap')
                tjs_rewards[tj_slice] = self.reward.data.take(bf_start + tj_indices, axis=0, mode='wrap')
                tjs_terminations[tj_slice] = self.terminate.data.take(bf_start + tj_indices, axis=0, mode='wrap')
                tjs_timeout[tj_slice] = self.timeout.data.take(bf_start + tj_indices, axis=0, mode='wrap')
                tjs_bprobabilities[tj_slice] = self.bprobabilities.data.take(bf_start + tj_indices, axis=0,
                                                                             mode='wrap')
                # Stacks of states
                trj_state_stack_sz = 1 + right_terminal_stop
                trj_state_stack = self.state.data.take(bf_start + start_idx + np.arange(trj_state_stack_sz), mode='wrap',
                                                       axis=0)
                trj_state_stack[:left_terminal_idx] *= 0

                state_stack_slices = np.arange(trj_state_stack_sz - 1 + 1)[:, None] \
                                     + np.arange(1)
                state_stacks = trj_state_stack.take(state_stack_slices, axis=0)

                sample_states[batch_idx] = state_stacks[0]
                tjs_states[tj_slice] = state_stacks[1:]

                computed_return_buffer_inds[batch_idx - retrieved_count] += idx
                computed_return_batch_inds[batch_idx - retrieved_count] += batch_idx
                tj_start_idx += self.n
                batch_idx += 1

        # We wait until the end to retrieve the q_values because it's more efficient to make only one call to
        # update_function when using a gpu.
        adjusted_batch_sz = self.batch_sz - retrieved_count
        tjs_states = np.squeeze(tjs_states[:adjusted_batch_sz*self.n]).reshape((adjusted_batch_sz * self.n,) +
                                                                               tuple(self.env_state_dims))
        tjs_qvalues = update_function(tjs_states, reshape=False).reshape([adjusted_batch_sz, self.n, self.num_actions])
        tjs_actions = tjs_actions[:adjusted_batch_sz*self.n].reshape([adjusted_batch_sz, self.n])
        tjs_rewards = tjs_rewards[:adjusted_batch_sz*self.n].reshape([adjusted_batch_sz, self.n])
        tjs_terminations = tjs_terminations[:adjusted_batch_sz*self.n].reshape([adjusted_batch_sz, self.n])
        tjs_timeout = tjs_timeout[:adjusted_batch_sz * self.n].reshape([adjusted_batch_sz, self.n])
        tjs_bprobabilities = tjs_bprobabilities[:adjusted_batch_sz*self.n].reshape([adjusted_batch_sz, self.n,
                                                                                    self.num_actions])

        computed_return_batch_inds = computed_return_batch_inds[:self.batch_sz-retrieved_count]
        estimated_returns[computed_return_batch_inds] = \
            self.return_function.batch_iterative_return_function(tjs_rewards, tjs_actions, tjs_qvalues,
                                                                 tjs_terminations, tjs_timeout,
                                                                 tjs_bprobabilities,
                                                                 adjusted_batch_sz)

        computed_return_buffer_inds = computed_return_buffer_inds[:self.batch_sz-retrieved_count]
        self.estimated_return.data.put(indices=bf_start + computed_return_buffer_inds,
                                       values=estimated_returns[computed_return_batch_inds], mode='wrap')
        self.up_to_date.data.put(indices=bf_start + computed_return_buffer_inds, values=True, mode='wrap')
        return sample_states, sample_actions, estimated_returns

    def ready_to_sample(self):
        return self.batch_sz < (self.current_index - (self.n + 1))

    def out_of_date(self):
        self.up_to_date.data[:] = False
