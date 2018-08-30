" Project Packages "
from Experiments_Engine.Objects_Bases import EnvironmentBase, FunctionApproximatorBase, PolicyBase

" Math packages "
from pylab import random, cos
import numpy as np

from Experiments_Engine.config import Config
from Experiments_Engine.Util import check_dict_else_default, check_attribute_else_default


class MountainCar(EnvironmentBase):
    """
    Environment Specifications:
    Number of Actions = 3
    Observation Dimension = 2 (position, velocity)
    Observation Dtype = np.float32
    Reward = -1 at every step

    Summary Name: steps_per_episode
    """

    def __init__(self, config=None, summary=None):

        super().__init__()
        assert isinstance(config, Config)
        """ Parameters:
        Name:                       Type            Default:        Description(omitted when self-explanatory):
        max_actions                 int             5000            The max number of actions executed before forcing
                                                                    termination
        save_summary                bool            False           Whether to save a summary of the environment
        """
        self.max_actions = check_attribute_else_default(config, 'max_actions', 5000)
        self.save_summary = check_attribute_else_default(config, 'save_summary', False)
        self.summary = summary
        if self.save_summary:
            assert isinstance(self.summary, dict)
            check_dict_else_default(self.summary, "steps_per_episode", [])

        " Inner state of the environment "
        self.step_count = 0
        self.current_state = self.reset()
        self.actions = np.array([0, 1, 2], dtype=int)  # 0 = backward, 1 = coast, 2 = forward
        self.high = np.array([0.5, 0.07], dtype=np.float32)
        self.low = np.array([-1.2, -0.07], dtype=np.float32)
        self.action_dictionary = {0: -1,    # accelerate backwards
                                   1: 0,    # coast
                                   2: 1}    # accelerate forwards

    def reset(self):
        # random() returns a random float in the half open interval [0,1)
        position = -0.6 + random() * 0.2
        velocity = 0.0
        self.current_state = np.array((position, velocity), dtype=np.float32)
        self.step_count = 0
        return self.current_state

    " Update environment "
    def update(self, A):
        self.step_count += 1

        if A not in self.actions:
            raise ValueError("The action should be one of the following integers: {0, 1, 2}.")
        action = self.action_dictionary[A]
        reward = -1.0
        terminate = False
        timeout = False

        if self.step_count >= self.max_actions:
            timeout = True

        current_position = self.current_state[0]
        current_velocity = self.current_state[1]

        velocity = current_velocity + (0.001 * action) - (0.0025 * cos(3 * current_position))
        position = current_position + velocity

        if velocity > 0.07:
            velocity = 0.07
        elif velocity < -0.07:
            velocity = -0.07

        if position < -1.2:
            position = -1.2
            velocity = 0.0
        elif position > 0.5:
            position = 0.5
            terminate = True

        if terminate:
            if self.save_summary:
                self.summary['steps_per_episode'].append(self.step_count)
            self.step_count = 0

        self.current_state = np.array((position, velocity), dtype=np.float64)

        return self.current_state, reward, terminate, timeout

    " Getters "
    def get_num_actions(self):
        return 3

    def get_actions(self):
        return self.actions

    def get_observation_dimensions(self):
        return [2]

    def get_current_state(self):
        return self.current_state

    def get_observation_dtype(self):
        return self.current_state.dtype

    def get_state_for_er_buffer(self):
        return self.current_state
