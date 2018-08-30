import numpy as np
import tensorflow as tf

from Experiments_Engine.Objects_Bases import FunctionApproximatorBase
from Experiments_Engine.config import Config
from Experiments_Engine.Util import check_attribute_else_default, check_dict_else_default

" Neural Network function approximator "
class NeuralNetwork_wER_FA(FunctionApproximatorBase):

    def __init__(self, optimizer, target_network, update_network, er_buffer, config=None, tf_session=None,
                 restore=False, summary=None):
        """
        Summary Names:
            cumulative_loss
            training_steps
        """

        super().__init__()
        assert isinstance(config, Config)
        self.config = config
        """ 
        Parameters in config:
        Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
        alpha                   float           0.00025             step size parameter
        obs_dims                list            [4,84,84]           the dimensions of the obsevations
        tnetwork_update_freq    int             10,000              number of updates before updating the target network
        update_count            int             0                   number of updates performed
        save_summary            bool            False               indicates whether to save a summary of training
        """
        self.alpha = check_attribute_else_default(self.config, 'alpha', 0.00025)
        self.obs_dims = check_attribute_else_default(self.config, 'obs_dims', [4, 84, 84])
        self.tnetwork_update_freq = check_attribute_else_default(self.config, 'tnetwork_update_freq', 10000)
        self.save_summary = check_attribute_else_default(self.config, 'save_summary', False)
        check_attribute_else_default(self.config, 'update_count', 0)
        self.summary = summary
        if self.save_summary:
            assert isinstance(self.summary, dict)
            check_dict_else_default(self.summary, 'cumulative_loss', [])
            check_dict_else_default(self.summary, 'training_steps', [])
            self.training_steps = 0
            self.cumulative_loss = 0

        """ Other Parameters """
        " Experience Replay Buffer and Return Function "
        self.er_buffer = er_buffer

        " Neural Network Models "
        self.target_network = target_network    # Target Network
        self.update_network = update_network    # Update Network

        " Training and Learning Evaluation: Tensorflow and variables initializer "
        self.optimizer = optimizer(self.alpha)
        self.sess = tf_session or tf.Session()

        " Train step "
        self.train_step = self.optimizer.minimize(self.update_network.train_loss,
                                                  var_list=self.update_network.train_vars[0])

        " Initializing variables in the graph"
        if not restore:
            for var in tf.global_variables():
                self.sess.run(var.initializer)

    def update(self, state=None, action=None, nstep_return=None):
        if self.er_buffer.ready_to_sample():
            sample_frames, sample_actions, sample_returns = self.er_buffer.get_data(update_function=self.get_next_states_values)
            sample_actions = np.column_stack([np.arange(len(sample_actions)), sample_actions])
            feed_dictionary = {self.update_network.x_frames: np.squeeze(sample_frames),
                               self.update_network.x_actions: sample_actions,
                               self.update_network.y: sample_returns}

            train_loss, _ = self.sess.run((self.update_network.train_loss, self.train_step), feed_dict=feed_dictionary)
            self.training_steps += 1
            self.cumulative_loss += train_loss
            self.config.update_count += 1
            if self.config.update_count >= self.tnetwork_update_freq:
                self.config.update_count = 0
                self.update_target_network()
                self.er_buffer.out_of_date()

    def update_target_network(self):
        update_network_vars = self.update_network.get_variables_as_tensor()
        self.target_network.replace_model_weights(new_vars=update_network_vars, tf_session=self.sess)

    def get_value(self, state, action):
        y_hat = self.get_next_states_values(state)
        return y_hat[action]

    def get_next_states_values(self, state, reshape=True):
        if reshape:
            dims = [1] + list(self.obs_dims)
            feed_dictionary = {self.target_network.x_frames: state.reshape(dims)}
            y_hat = self.sess.run(self.target_network.y_hat, feed_dict=feed_dictionary)
            return y_hat[0]
        else:
            feed_dictionary = {self.target_network.x_frames: state}
            y_hat = self.sess.run(self.target_network.y_hat, feed_dict=feed_dictionary)
            return y_hat

    def store_in_summary(self):
        if self.save_summary:
            self.summary['cumulative_loss'].append(self.cumulative_loss)
            self.summary['training_steps'].append(self.training_steps)
            self.cumulative_loss = 0
            self.training_steps = 0

