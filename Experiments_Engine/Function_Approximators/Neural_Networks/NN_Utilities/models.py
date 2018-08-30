import tensorflow as tf
import numpy as np

from Experiments_Engine.Function_Approximators.Neural_Networks.NN_Utilities import layers
from Experiments_Engine.Objects_Bases.NN_Model_Base import ModelBase
from Experiments_Engine.config import Config
from Experiments_Engine.Util import check_attribute_else_default


def linear_transfer(x):
    return x

"""
Creates a model with n convolutinal layers followed by a pooling step and m fully connected layers followed by
one linear output layer
"""
class Model_nCPmFO(ModelBase):

    def __init__(self, config=None, name="default", SEED=None):
        super().__init__()

        assert isinstance(config, Config)
        """ 
        Parameters in config:
        Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
        dim_out                 list            [10,10,10]          the output dimensions of each layer, i.e. neurons
        filter_dims             list            [2,2]               the dimensions of each filter
        strides                 list            [4, 2]              strides use by each convolutional layer
        obs_dims                list            [4,84,84]           the dimensions of the observations seen by the agent
        num_actions             int             2                   the number of actions available to the agent
        gate_fun                tf gate fun     tf.nn.relu          the gate function used across the whole network
        conv_layers             int             2                   number of convolutional layers
        full_layers             int             1                   number of fully connected layers
        max_pool                bool            True                indicates whether to max pool between each conv layer
        frames_format           str             "NCHW"              Specifies the format of the frames fed to the network
        norm_factor             float           1                   Normalizes the frames by the value provided               
        """
        self.dim_out = check_attribute_else_default(config, 'dim_out', [10,10,10])
        self.filter_dims = check_attribute_else_default(config, 'filter_dims', [2,2])
        self.strides = check_attribute_else_default(config, 'strides', [4,2])
        channels, height, width = check_attribute_else_default(config, 'obs_dims', [4, 84, 84])
        num_actions = check_attribute_else_default(config, 'num_actions', 2)
        self.gate_fun = check_attribute_else_default(config, 'gate_fun', tf.nn.relu)
        self.convolutional_layers = check_attribute_else_default(config, 'conv_layers', 2)
        self.fully_connected_layers = check_attribute_else_default(config, 'full_layers', 1)
        self.max_pool = check_attribute_else_default(config, 'max_pool', True)
        self.frames_format = check_attribute_else_default(config, 'frames_format', 'NCHW')
        self.norm_factor = check_attribute_else_default(config, 'norm_factor', 1.)

        """
        Other Parameters:
        name - name of the network. Should be a string.
        """
        self.name = name
        row_and_action_number = 2
        total_layers = self.convolutional_layers + self.fully_connected_layers

        " Placehodler "
        self.x_frames = tf.placeholder(tf.float32, shape=(None, channels, height, width))   # input frames
        self.x_frames = tf.divide(self.x_frames, self.norm_factor)
        self.x_actions = tf.placeholder(tf.int32, shape=(None, row_and_action_number))      # input actions
        self.y = tf.placeholder(tf.float32, shape=None)                                     # target

        " Variables for Training "
        self.train_vars = []

        """ Convolutional layers """
        dim_in_conv = [channels] + self.dim_out[:self.convolutional_layers - 1]
        current_s_hat = self.x_frames
        if self.frames_format == "NHWC":
            current_s_hat = tf.transpose(current_s_hat, [0, 2, 3, 1])
        for i in range(self.convolutional_layers):
            # layer n: convolutional
            W, b, z_hat, r_hat = layers.convolution_2d(
                self.name, "conv_"+str(i+1), current_s_hat, self.filter_dims[i], dim_in_conv[i], self.dim_out[i],
                tf.random_normal_initializer(stddev=1.0 / np.sqrt(self.filter_dims[i]**2 * dim_in_conv[i] + 1),
                                             seed=SEED), self.gate_fun, stride=self.strides[i],
                format=self.frames_format)
            # layer n + 1/2: pool
            if self.max_pool:
                s_hat = tf.nn.max_pool(
                    r_hat, ksize=[1, 1, 2, 2], strides=[1, 1, 2, 2], padding="SAME")
            else:
                s_hat = r_hat

            current_s_hat = s_hat
            self.train_vars.extend([W, b])

        """ Fully Connected layers """
        shape = current_s_hat.get_shape().as_list()
        current_y_hat = tf.reshape(current_s_hat, [-1, shape[1] * shape[2] * shape[3]])
        # shape[-3:] are the last 3 dimensions. Shape has 4 dimensions: dim 1 = None, dim 2 =
        dim_in_fully = [np.prod(shape[-3:])] + self.dim_out[self.convolutional_layers: total_layers-1]
        dim_out_fully = self.dim_out[self.convolutional_layers:]
        for j in range(self.fully_connected_layers):
            # layer n + m: fully connected
            W, b, z_hat, y_hat = layers.fully_connected(
                self.name, "full_"+str(j+1), current_y_hat, dim_in_fully[j], dim_out_fully[j],
                tf.random_normal_initializer(stddev=1.0 / np.sqrt(dim_in_fully[j]), seed=SEED), self.gate_fun)

            current_y_hat = y_hat
            self.train_vars.extend([W, b])

        """ Output layer """
        # output layer: fully connected
        W, b, z_hat, self.y_hat = layers.fully_connected(
            self.name, "output_layer", current_y_hat, self.dim_out[-1], num_actions,
            tf.random_normal_initializer(stddev=1.0 / np.sqrt(self.dim_out[-1]), seed=SEED), linear_transfer)
        self.train_vars.extend([W, b])
        self.train_vars = [self.train_vars]

        # Obtaining y_hat and Scaling by the Importance Sampling
        y_hat = tf.gather_nd(self.y_hat, self.x_actions)
        y = self.y
        # Temporal Difference Error
        self.td_error = tf.subtract(y, y_hat)
        # Loss
        self.train_loss = tf.reduce_sum(tf.pow(self.td_error, 2))

    def replace_model_weights(self, new_vars, tf_session=tf.Session()):
        if not isinstance(new_vars, list):
            new_vars = [new_vars]
        assert len(new_vars) == len(self.train_vars[0]), "The lists of variables need to have the same length!"

        for i in range(len(self.train_vars[0])):
            tf_session.run(tf.assign(self.train_vars[0][i], new_vars[i]))

    def get_variables_as_list(self, tf_session=tf.Session()):
        var_list = []
        for i in range(len(self.train_vars[0])):
            var_list.append(tf_session.run(self.train_vars[0][i]))
        return var_list

    def get_variables_as_tensor(self):
        return self.train_vars[0]


"""
Creates a model with m fully connected layers followed by one linear output layer
"""
class Model_mFO(ModelBase):

    def __init__(self, config=None, name="default", SEED=None):
        super().__init__()

        assert isinstance(config, Config)
        """ 
        Parameters in config:
        Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
        dim_out                 list            [10,10,10]          the output dimensions of each layer, i.e. neurons
        obs_dims                list            [2]                 the dimensions of the observations seen by the agent
        num_actions             int             3                   the number of actions available to the agent
        gate_fun                tf gate fun     tf.nn.relu          the gate function used across the whole network
        full_layers             int             3                   number of fully connected layers
        """
        self.dim_out = check_attribute_else_default(config, 'dim_out', [10,10,10])
        self.obs_dims = check_attribute_else_default(config, 'obs_dims', [2])
        self.num_actions = check_attribute_else_default(config, 'num_actions', 3)
        self.gate_fun = check_attribute_else_default(config, 'gate_fun', tf.nn.relu)
        self.full_layers = check_attribute_else_default(config, 'full_layers', 3)

        """
        Other Parameters:
        name - name of the network. Should be a string.
        """
        self.name = name

        " Dimensions "
        dim_in = [np.prod(self.obs_dims)] + self.dim_out[:-1]
        row_and_action_number = 2
        " Placehodler "
        self.x_frames = tf.placeholder(tf.float32, shape=(None, dim_in[0]))             # input frames
        self.x_actions = tf.placeholder(tf.int32, shape=(None, row_and_action_number))  # input actions
        self.y = tf.placeholder(tf.float32, shape=None)                                 # target
        " Variables for Training "
        self.train_vars = []

        " Fully Connected Layers "
        current_y_hat = self.x_frames
        for j in range(self.full_layers):
            # layer n + m: fully connected
            W, b, z_hat, y_hat = layers.fully_connected(
                self.name, "full_" + str(j + 1), current_y_hat, dim_in[j], self.dim_out[j],
                tf.random_normal_initializer(stddev=1.0 / np.sqrt(dim_in[j]), seed=SEED), self.gate_fun)

            current_y_hat = y_hat
            self.train_vars.extend([W, b])

        """ Output layer """
        # output layer: fully connected
        W, b, z_hat, self.y_hat = layers.fully_connected(
            self.name, "output_layer", current_y_hat, self.dim_out[-1], self.num_actions,
            tf.random_normal_initializer(stddev=1.0 / np.sqrt(self.dim_out[-1]), seed=SEED), linear_transfer)
        self.train_vars.extend([W, b])
        self.train_vars = [self.train_vars]

        # Obtaining y_hat and Scaling by the Importance Sampling
        y_hat = tf.gather_nd(self.y_hat, self.x_actions)
        y = self.y
        # Temporal Difference Error
        self.td_error = tf.subtract(y, y_hat)
        # Loss
        self.train_loss = tf.reduce_sum(tf.pow(self.td_error, 2))

    def replace_model_weights(self, new_vars, tf_session=tf.Session()):
        if not isinstance(new_vars, list):
            new_vars = [new_vars]
        assert len(new_vars) == len(self.train_vars[0]), "The lists of variables need to have the same length!"

        for i in range(len(self.train_vars[0])):
            tf_session.run(tf.assign(self.train_vars[0][i], new_vars[i]))

    def get_variables_as_list(self, tf_session=tf.Session()):
        var_list = []
        for i in range(len(self.train_vars[0])):
            var_list.append(tf_session.run(self.train_vars[0][i]))
        return var_list

    def get_variables_as_tensor(self):
        return self.train_vars[0]


""" Experimental Stuff """
"""
Creates a model with n convolutinal layers followed by a pooling step and m fully connected layers followed by
one linear output layer. The last fully connected layer of the network uses a radial basis function of the form
f(x) = e^{(x-c)^2 / (1/(# of neurons)), where c is a uniform sample from [0,1] and fixed for the rest of training.
"""
class Model_nCPmFO_wRBFLayer2(ModelBase):

    def __init__(self, config=None, name="default", SEED=None, centers=None, stddevs=None):
        super().__init__()

        assert isinstance(config, Config)
        """ 
        Parameters in config:
        Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
        dim_out                 list            [10,10,10]          the output dimensions of each layer, i.e. neurons
        filter_dims             list            [2,2]               the dimensions of each filter
        strides                 list            [4, 2]              strides use by each convolutional layer
        obs_dims                list            [4,84,84]           the dimensions of the observations seen by the agent
        num_actions             int             2                   the number of actions available to the agent
        gate_fun                tf gate fun     tf.nn.relu          the gate function used across the whole network
        conv_layers             int             2                   number of convolutional layers
        full_layers             int             1                   number of fully connected layers
        max_pool                bool            True                indicates whether to max pool between each conv layer
        frames_format           str             "NCHW"              Specifies the format of the frames fed to the network
        norm_factor             float           1                   Normalizes the frames by the value provided               
        """
        self.dim_out = check_attribute_else_default(config, 'dim_out', [10,10,10])
        self.filter_dims = check_attribute_else_default(config, 'filter_dims', [2,2])
        self.strides = check_attribute_else_default(config, 'strides', [4,2])
        channels, height, width = check_attribute_else_default(config, 'obs_dims', [4, 84, 84])
        num_actions = check_attribute_else_default(config, 'num_actions', 2)
        self.gate_fun = check_attribute_else_default(config, 'gate_fun', tf.nn.relu)
        self.convolutional_layers = check_attribute_else_default(config, 'conv_layers', 2)
        self.fully_connected_layers = check_attribute_else_default(config, 'full_layers', 1)
        self.max_pool = check_attribute_else_default(config, 'max_pool', True)
        self.frames_format = check_attribute_else_default(config, 'frames_format', 'NCHW')
        self.norm_factor = check_attribute_else_default(config, 'norm_factor', 1.)

        """
        Other Parameters:
        name - name of the network. Should be a string.
        """
        self.name = name
        row_and_action_number = 2
        total_layers = self.convolutional_layers + self.fully_connected_layers

        " Placehodler "
        self.x_frames = tf.placeholder(tf.float32, shape=(None, channels, height, width))   # input frames
        self.x_frames = tf.divide(self.x_frames, self.norm_factor)
        self.x_actions = tf.placeholder(tf.int32, shape=(None, row_and_action_number))      # input actions
        self.y = tf.placeholder(tf.float32, shape=None)                                     # target

        " Variables for Training "
        self.train_vars = []

        """ Convolutional layers """
        dim_in_conv = [channels] + self.dim_out[:self.convolutional_layers - 1]
        current_s_hat = self.x_frames
        if self.frames_format == "NHWC":
            current_s_hat = tf.transpose(current_s_hat, [0, 2, 3, 1])

        centers = tf.constant(np.random.uniform(low=-1, high=1,size=(self.dim_out[0], 21, 21)), dtype=np.float32)
        stddev = tf.constant(tf.divide(1, np.prod((self.dim_out[0], 21, 21))), dtype=np.float32)
        W, b, z_hat, r_hat = layers.convolution_2d_rbf(self.name, 'conv_rbf_1', current_s_hat, self.filter_dims[0],
                                                       dim_in_conv[0], self.dim_out[0],
                tf.random_normal_initializer(stddev=1.0/np.sqrt(self.filter_dims[0]**2 * dim_in_conv[0] +1), seed=SEED),
                                                       stride=self.strides[0], center=centers, stddev=stddev,
                                                       format=self.frames_format)
        if self.max_pool:
            s_hat = tf.nn.max_pool(r_hat, ksize=[1,1,2,2], strides=[1,1,2,2], padding='SAME')
        else:
            s_hat = r_hat
        current_s_hat = s_hat
        self.train_vars.extend([W,b])

        for i in range(1, self.convolutional_layers):
            # layer n: convolutional
            W, b, z_hat, r_hat = layers.convolution_2d(
                self.name, "conv_"+str(i+1), current_s_hat, self.filter_dims[i], dim_in_conv[i], self.dim_out[i],
                tf.random_normal_initializer(stddev=1.0 / np.sqrt(self.filter_dims[i]**2 * dim_in_conv[i] + 1),
                                             seed=SEED), self.gate_fun, stride=self.strides[i],
                format=self.frames_format)
            # layer n + 1/2: pool
            if self.max_pool:
                s_hat = tf.nn.max_pool(
                    r_hat, ksize=[1, 1, 2, 2], strides=[1, 1, 2, 2], padding="SAME")
            else:
                s_hat = r_hat

            current_s_hat = s_hat
            self.train_vars.extend([W, b])

        """ Fully Connected layers """
        shape = current_s_hat.get_shape().as_list()
        current_y_hat = tf.reshape(current_s_hat, [-1, shape[1] * shape[2] * shape[3]])
        # shape[-3:] are the last 3 dimensions. Shape has 4 dimensions: dim 1 = None, dim 2 =
        dim_in_fully = [np.prod(shape[-3:])] + self.dim_out[self.convolutional_layers: total_layers-1]
        dim_out_fully = self.dim_out[self.convolutional_layers:]
        for j in range(self.fully_connected_layers-1):
            # layer n + m: fully connected
            W, b, z_hat, y_hat = layers.fully_connected(
                self.name, "full_"+str(j+1), current_y_hat, dim_in_fully[j], dim_out_fully[j],
                tf.random_normal_initializer(stddev=1.0 / np.sqrt(dim_in_fully[j]), seed=SEED), self.gate_fun)

            current_y_hat = y_hat
            self.train_vars.extend([W, b])

        centers = tf.constant(np.random.uniform(low=-1, high=1,size=(dim_in_fully[-1],dim_out_fully[-1])), dtype=np.float32)
        stddev = tf.constant(tf.divide(1, np.prod((dim_in_fully[-1], dim_out_fully[-1]))), dtype=np.float32)
        W, b, z_hat, y_hat = layers.fully_connected_rbf(self.name, 'full_rbf', current_y_hat, dim_in_fully[-1],
            dim_out_fully[-1], tf.random_normal_initializer(stddev=1.0 / np.sqrt(dim_in_fully[-1]), seed=SEED),
            center=centers, stddev=stddev)
        current_y_hat = y_hat
        self.train_vars.extend([W,b])

        """ Output layer """
        # output layer: fully connected
        W, b, z_hat, self.y_hat = layers.fully_connected(
            self.name, "output_layer", current_y_hat, self.dim_out[-1], num_actions,
            tf.random_normal_initializer(stddev=1.0 / np.sqrt(self.dim_out[-1]), seed=SEED), linear_transfer)
        self.train_vars.extend([W, b])
        self.train_vars = [self.train_vars]

        # Obtaining y_hat and Scaling by the Importance Sampling
        y_hat = tf.gather_nd(self.y_hat, self.x_actions)
        y = self.y
        # Temporal Difference Error
        self.td_error = tf.subtract(y, y_hat)
        # Loss
        self.train_loss = tf.reduce_sum(tf.pow(self.td_error, 2))

    def replace_model_weights(self, new_vars, tf_session=tf.Session()):
        if not isinstance(new_vars, list):
            new_vars = [new_vars]
        assert len(new_vars) == len(self.train_vars[0]), "The lists of variables need to have the same length!"

        for i in range(len(self.train_vars[0])):
            tf_session.run(tf.assign(self.train_vars[0][i], new_vars[i]))

    def get_variables_as_list(self, tf_session=tf.Session()):
        var_list = []
        for i in range(len(self.train_vars[0])):
            var_list.append(tf_session.run(self.train_vars[0][i]))
        return var_list

    def get_variables_as_tensor(self):
        return self.train_vars[0]


"""
Creates a model with n convolutinal layers followed by a pooling step and m fully connected layers followed by
one linear output layer. All the layers of the network use a radial basis function of the form
f(x) = e^{(x-c)^2 / (1/(# of neurons)), where c is a uniform sample from [-1,1] and fixed for the rest of training and
x is normalized as norm(x_i) = x_i / abs(max(X)) where X is a matrix and the max is taken over all the rows and columns
"""
class Model_nCPmFO_wRBFLayer(ModelBase):

    def __init__(self, config=None, name="default", SEED=None):
        super().__init__()

        assert isinstance(config, Config)
        """ 
        Parameters in config:
        Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
        dim_out                 list            [10,10,10]          the output dimensions of each layer, i.e. neurons
        filter_dims             list            [2,2]               the dimensions of each filter
        strides                 list            [4, 2]              strides use by each convolutional layer
        obs_dims                list            [4,84,84]           the dimensions of the observations seen by the agent
        num_actions             int             2                   the number of actions available to the agent
        gate_fun                tf gate fun     tf.nn.relu          the gate function used across the whole network
        conv_layers             int             2                   number of convolutional layers
        full_layers             int             1                   number of fully connected layers
        max_pool                bool            True                indicates whether to max pool between each conv layer
        frames_format           str             "NCHW"              Specifies the format of the frames fed to the network
        norm_factor             float           1                   Normalizes the frames by the value provided               
        """
        self.dim_out = check_attribute_else_default(config, 'dim_out', [10,10,10])
        self.filter_dims = check_attribute_else_default(config, 'filter_dims', [2,2])
        self.strides = check_attribute_else_default(config, 'strides', [4,2])
        channels, height, width = check_attribute_else_default(config, 'obs_dims', [4, 84, 84])
        num_actions = check_attribute_else_default(config, 'num_actions', 2)
        self.gate_fun = check_attribute_else_default(config, 'gate_fun', tf.nn.relu)
        self.convolutional_layers = check_attribute_else_default(config, 'conv_layers', 2)
        self.fully_connected_layers = check_attribute_else_default(config, 'full_layers', 1)
        self.max_pool = check_attribute_else_default(config, 'max_pool', True)
        self.frames_format = check_attribute_else_default(config, 'frames_format', 'NCHW')
        self.norm_factor = check_attribute_else_default(config, 'norm_factor', 1.)

        """
        Other Parameters:
        name - name of the network. Should be a string.
        """
        self.name = name
        row_and_action_number = 2
        total_layers = self.convolutional_layers + self.fully_connected_layers

        " Placehodler "
        self.x_frames = tf.placeholder(tf.float32, shape=(None, channels, height, width))   # input frames
        self.x_frames = tf.divide(self.x_frames, self.norm_factor)
        self.x_actions = tf.placeholder(tf.int32, shape=(None, row_and_action_number))      # input actions
        self.y = tf.placeholder(tf.float32, shape=None)                                     # target

        " Variables for Training "
        self.train_vars = []

        """ Convolutional layers """
        dim_in_conv = [channels] + self.dim_out[:self.convolutional_layers - 1]
        current_s_hat = self.x_frames
        if self.frames_format == "NHWC":
            current_s_hat = tf.transpose(current_s_hat, [0, 2, 3, 1])

        for i in range(self.convolutional_layers):
            if self.frames_format == "NHWC":
                out_height = np.ceil(current_s_hat.shape[1]._value / self.strides[i])
                out_width = np.ceil(current_s_hat.shape[2]._value / self.strides[i])
                centers_shape = np.array((out_height, out_width, self.dim_out[i]), dtype=np.uint32)
            else: # Format = "NCHW"
                out_height = np.ceil(current_s_hat.shape[2]._value / self.strides[i])
                out_width = np.ceil(current_s_hat.shape[3]._value /self.strides[i])
                centers_shape = np.array((self.dim_out[i], out_height, out_width), dtype=np.uint32)
            centers = tf.constant(np.random.uniform(0,1, size=centers_shape), dtype=tf.float32)
            stddev = tf.constant(1/self.dim_out[i], dtype=tf.float32)
            # layer n: convolutional
            W, b, z_hat, r_hat = layers.convolution_2d_rbf(
                self.name, "conv_rbf_"+str(i+1), current_s_hat, self.filter_dims[i], dim_in_conv[i], self.dim_out[i],
                tf.random_normal_initializer(stddev=1.0 / np.sqrt(self.filter_dims[i]**2 * dim_in_conv[i] + 1),
                                             seed=SEED),
                center=centers, stddev=stddev, stride=self.strides[i], format=self.frames_format)
            # layer n + 1/2: pool
            if self.max_pool:
                s_hat = tf.nn.max_pool(
                    r_hat, ksize=[1, 1, 2, 2], strides=[1, 1, 2, 2], padding="SAME")
            else:
                s_hat = r_hat

            current_s_hat = s_hat
            self.train_vars.extend([W, b])

        """ Fully Connected layers """
        shape = current_s_hat.get_shape().as_list()
        current_y_hat = tf.reshape(current_s_hat, [-1, shape[1] * shape[2] * shape[3]])
        # shape[-3:] are the last 3 dimensions. Shape has 4 dimensions: dim 1 = None, dim 2 =
        dim_in_fully = [np.prod(shape[-3:])] + self.dim_out[self.convolutional_layers: total_layers-1]
        dim_out_fully = self.dim_out[self.convolutional_layers:]
        for j in range(self.fully_connected_layers):
            centers_shape = (dim_in_fully[j], dim_out_fully[j])
            centers = tf.constant(np.random.uniform(low=0, high=1, size=centers_shape), dtype=tf.float32)
            stddev = tf.constant(1/dim_out_fully[j], dtype=np.float32)

            # layer n + m: fully connected
            W, b, z_hat, y_hat = layers.fully_connected_rbf(
                self.name, "full_rbf_"+str(j+1), current_y_hat, dim_in_fully[j], dim_out_fully[j],
                tf.random_normal_initializer(stddev=1.0 / np.sqrt(dim_in_fully[j]), seed=SEED),
                center=centers, stddev=stddev)

            current_y_hat = y_hat
            self.train_vars.extend([W, b])

        """ Output layer """
        # output layer: fully connected
        W, b, z_hat, self.y_hat = layers.fully_connected(
            self.name, "output_layer", current_y_hat, self.dim_out[-1], num_actions,
            tf.random_normal_initializer(stddev=1.0 / np.sqrt(self.dim_out[-1]), seed=SEED), linear_transfer)
        self.train_vars.extend([W, b])
        self.train_vars = [self.train_vars]

        # Obtaining y_hat and Scaling by the Importance Sampling
        y_hat = tf.gather_nd(self.y_hat, self.x_actions)
        y = self.y
        # Temporal Difference Error
        self.td_error = tf.subtract(y, y_hat)
        # Loss
        self.train_loss = tf.reduce_sum(tf.pow(self.td_error, 2))

    def replace_model_weights(self, new_vars, tf_session=tf.Session()):
        if not isinstance(new_vars, list):
            new_vars = [new_vars]
        assert len(new_vars) == len(self.train_vars[0]), "The lists of variables need to have the same length!"

        for i in range(len(self.train_vars[0])):
            tf_session.run(tf.assign(self.train_vars[0][i], new_vars[i]))

    def get_variables_as_list(self, tf_session=tf.Session()):
        var_list = []
        for i in range(len(self.train_vars[0])):
            var_list.append(tf_session.run(self.train_vars[0][i]))
        return var_list

    def get_variables_as_tensor(self):
        return self.train_vars[0]

########################################################################################################################
##########################                    Old Code                            ######################################
########################################################################################################################
"""
Creates a model with n convolutinal layers followed by a pooling step and m fully connected layers for positive
and negative returns followed by one linear output layer that combines both networks together
"""
class Model_nCPmFO_RP(ModelBase):
    def __init__(self, name=None, dim_out=None, filter_dims=None, observation_dimensions=None, num_actions=None,
                 gate_fun=None, convolutional_layers=None, fully_connected_layers=None, SEED=None,
                 model_dictionary=None, eta=1.0, reward_path=False):
        super().__init__()
        if model_dictionary is None:
            self._model_dictionary = {"model_name": name,
                                      "output_dims": dim_out,
                                      "filter_dims": filter_dims,
                                      "observation_dimensions": observation_dimensions,
                                      "num_actions": num_actions,
                                      "gate_fun": gate_fun,
                                      "conv_layers": convolutional_layers,
                                      "full_layers": fully_connected_layers,
                                      "eta": eta,
                                      "reward_path": reward_path}
        else:
            self._model_dictionary = model_dictionary
        " Loading Variables From Dictionary "
        eta = self._model_dictionary["eta"]
        fully_connected_layers = self._model_dictionary["full_layers"]
        convolutional_layers = self._model_dictionary["conv_layers"]
        name = self._model_dictionary["model_name"]
        dim_out = self._model_dictionary["output_dims"]
        gate_fun = self._model_dictionary["gate_fun"]
        filter_dims = self._model_dictionary["filter_dims"]
        reward_path = self._model_dictionary["reward_path"]
        " Reward Path Flag "
        if reward_path:
            train_vars_dims = 2
        else:
            train_vars_dims = 1
        " Dimensions "
        height, width, channels = self._model_dictionary["observation_dimensions"]
        actions = self._model_dictionary["num_actions"]
        row_and_action_number = 2
        total_layers = convolutional_layers + fully_connected_layers
        " Placehodler "
        self.x_frames = tf.placeholder(tf.float32, shape=(None, height, width, channels))   # input frames
        self.x_actions = tf.placeholder(tf.int32, shape=(None, row_and_action_number))      # input actions
        self.y = tf.placeholder(tf.float32, shape=None)                                     # target
        self.isampling = tf.placeholder(tf.float32, shape=None)                             # importance sampling term
        " Variables for Training "
        self.train_vars = []
        y_hats = []

        for k in range(train_vars_dims):
            """ Convolutional layers """
            temp_train_vars = []
            dim_in_conv = [channels] + dim_out[k][:convolutional_layers - 1]
            current_s_hat = self.x_frames
            for i in range(convolutional_layers):
                # layer n: convolutional
                W, b, z_hat, r_hat = layers.convolution_2d(
                    name, "conv_"+str(i+1)+"_"+str(k), current_s_hat, filter_dims[i], dim_in_conv[i], dim_out[k][i],
                    tf.random_normal_initializer(stddev=1.0 / np.sqrt(filter_dims[i]**2 * dim_in_conv[i] + 1), seed=SEED),
                    gate_fun)
                # layer n + 1/2: pool
                s_hat = tf.nn.max_pool(
                    r_hat, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

                current_s_hat = s_hat
                temp_train_vars.extend([W, b])

            """ Fully Connected layers """
            shape = current_s_hat.get_shape().as_list()
            current_y_hat = tf.reshape(current_s_hat, [-1, shape[1] * shape[2] * shape[3]])
            # shape[-3:] are the last 3 dimensions. Shape has 4 dimensions: dim 1 = None, dim 2 =
            dim_in_fully = [np.prod(shape[-3:])] + dim_out[k][convolutional_layers: total_layers-1]
            dim_out_fully = dim_out[k][convolutional_layers:]
            for j in range(fully_connected_layers):
                # layer n + m: fully connected
                W, b, z_hat, y_hat = layers.fully_connected(
                    name, "full_"+str(j+1)+"_"+str(k), current_y_hat, dim_in_fully[j], dim_out_fully[j],
                    tf.random_normal_initializer(stddev=1.0 / np.sqrt(dim_in_fully[j]), seed=SEED), gate_fun)

                current_y_hat = y_hat
                temp_train_vars.extend([W, b])

            y_hats.append(current_y_hat)
            self.train_vars.append(temp_train_vars)

        combined_y_hat = tf.concat(y_hats, 1)

        """ Output layer """
        # output layer: fully connected
        if reward_path:
            final_dim_in = dim_out[0][-1] + dim_out[1][-1]
        else:
            final_dim_in = dim_out[0][-1]
        W, b, z_hat, self.y_hat = layers.fully_connected(
            name, "output_layer", combined_y_hat,final_dim_in, actions,
            tf.random_normal_initializer(stddev=1.0 / np.sqrt(final_dim_in), seed=SEED), linear_transfer)
        for lst in self.train_vars:
            lst.extend([W, b])

        # Obtaining y_hat and Scaling by the Importance Sampling
        # y_hat = tf.gather_nd(self.y_hat, self.x_actions)
        y_hat = tf.multiply(self.y_hat, self.isampling)
        y = tf.multiply(self.y, self.isampling)
        # Temporal Difference Error
        self.td_error = tf.subtract(y_hat, y)
        self.squared_td_error = tf.reduce_sum(tf.pow(self.td_error, 2))

        # Regularizer
        regularizer = 0
        for lst in self.train_vars:
            for variable in lst:
                regularizer += tf.nn.l2_loss(variable)

        # Loss
        self.train_loss = self.squared_td_error + (eta * regularizer)

"""
Creates a model with m fully connected layers followed by one linear output layer that combines both networks together
"""
class Model_mFO_RP(ModelBase):
    def __init__(self, name=None, dim_out=None, observation_dimensions=None, num_actions=None, gate_fun=None,
                 fully_connected_layers=None, SEED=None, model_dictionary=None, eta=1.0, reward_path=False):
        super().__init__()
        if model_dictionary is None:
            self._model_dictionary = {"model_name": name,
                                      "output_dims": dim_out,
                                      "observation_dimensions": observation_dimensions,
                                      "num_actions": num_actions,
                                      "gate_fun": gate_fun,
                                      "full_layers": fully_connected_layers,
                                      "eta": eta,
                                      "reward_path": reward_path}
        else:
            self._model_dictionary = model_dictionary
        " Loading Variables From Dictionary "
        eta = self._model_dictionary["eta"]
        fully_connected_layers = self._model_dictionary["full_layers"]
        name = self._model_dictionary["model_name"]
        dim_out = self._model_dictionary["output_dims"]
        gate_fun = self._model_dictionary["gate_fun"]
        reward_path = self._model_dictionary["reward_path"]
        " Reward Path Flag "
        if reward_path:
            train_vars_dims = 2
        else:
            train_vars_dims = 1
        " Dimensions "
        dim_in = []
        for i in range(train_vars_dims):
            di = [np.prod(self._model_dictionary["observation_dimensions"])] \
                 + self._model_dictionary["output_dims"][i][:-1]
            dim_in.append(di)
        actions = self._model_dictionary["num_actions"]
        row_and_action_number = 2
        " Placehodler "
        self.x_frames = tf.placeholder(tf.float32,                                      # input frames
                                       shape=(None, np.prod(self._model_dictionary["observation_dimensions"])))
        self.x_actions = tf.placeholder(tf.int32, shape=(None, row_and_action_number))  # input actions
        self.y = tf.placeholder(tf.float32, shape=None)                                 # target
        self.isampling = tf.placeholder(tf.float32, shape=None)                         # importance sampling term
        " Variables for Training "
        self.train_vars = []
        y_hats = []

        for i in range(train_vars_dims):
            " Fully Connected Layers "
            train_vars = []
            current_y_hat = self.x_frames
            for j in range(fully_connected_layers):
                # layer n + m: fully connected
                W, b, z_hat, y_hat = layers.fully_connected(
                    name, "full_"+str(j + 1)+"_"+str(i), current_y_hat, dim_in[i][j], dim_out[i][j],
                    tf.random_normal_initializer(stddev=1.0 / np.sqrt(dim_in[i][j]), seed=SEED), gate_fun)

                current_y_hat = y_hat
                train_vars.extend([W, b])
            y_hats.append(current_y_hat)
            self.train_vars.append(train_vars)

        combined_y_hat = tf.concat(y_hats, 1)

        """ Output layer """
        # output layer: fully connected
        if reward_path:
            final_dim_in = dim_out[0][-1] + dim_out[1][-1]
        else:
            final_dim_in = dim_out[0][-1]
        W, b, z_hat, self.y_hat = layers.fully_connected(
            name, "output_layer", combined_y_hat, final_dim_in, actions,
            tf.random_normal_initializer(stddev=1.0 / np.sqrt(final_dim_in), seed=SEED), linear_transfer)
        for lst in self.train_vars:
            lst.extend([W, b])

        # Obtaining y_hat and Scaling by the Importance Sampling
        y_hat = tf.gather_nd(self.y_hat, self.x_actions)
        y_hat = tf.multiply(y_hat, self.isampling)
        y = tf.multiply(self.y, self.isampling)
        # Temporal Difference Error
        self.td_error = tf.subtract(y_hat, y)
        self.squared_td_error = tf.reduce_sum(tf.pow(self.td_error, 2))

        # Regularizer
        regularizer = 0
        for lst in self.train_vars:
            for variable in lst:
                regularizer += tf.nn.l2_loss(variable)

        # Loss
        self.train_loss = self.squared_td_error + (eta * regularizer)


"""
Creates a model with m fully connected layers followed by one fully connected layer with dropoconnect and
one fully connected linear unit
"""
class Model_mFO_RP_wDC(ModelBase):
    def __init__(self, name=None, dim_out=None, observation_dimensions=None, num_actions=None, gate_fun=None,
                 fully_connected_layers=None, SEED=None, model_dictionary=None, reward_path=False):
        super().__init__()

        " Model dictionary for saving and restoring "
        if model_dictionary is None:
            self._model_dictionary = {"model_name": name,
                                      "output_dims": dim_out,
                                      "observation_dimensions": observation_dimensions,
                                      "num_actions": num_actions,
                                      "gate_fun": gate_fun,
                                      "full_layers": fully_connected_layers,
                                      "reward_path": reward_path}
        else:
            self._model_dictionary = model_dictionary

        " Loading Variables From Dictionary "
        fully_connected_layers = self._model_dictionary["full_layers"]
        name = self._model_dictionary["model_name"]
        dim_out = self._model_dictionary["output_dims"]
        gate_fun = self._model_dictionary["gate_fun"]
        reward_path = self._model_dictionary["reward_path"]

        " Reward Path Flag "
        if reward_path:
            train_vars_dims = 2
        else:
            train_vars_dims = 1

        " Dimensions "
        dim_in = []
        for i in range(train_vars_dims):
            di = [np.prod(self._model_dictionary["observation_dimensions"])] \
                 + self._model_dictionary["output_dims"][i][:-1]
            dim_in.append(di)
        actions = self._model_dictionary["num_actions"]
        row_and_action_number = 2

        " Placehodler "
        self.x_frames = tf.placeholder(tf.float32,                                      # input frames
                                       shape=(None, np.prod(self._model_dictionary["observation_dimensions"])))
        self.x_actions = tf.placeholder(tf.int32, shape=(None, row_and_action_number))  # input actions
        self.y = tf.placeholder(tf.float32, shape=None)                                 # target
        self.isampling = tf.placeholder(tf.float32, shape=None)                         # importance sampling term
        " Variables for Training "
        self.train_vars = []
        y_hats = []

        for i in range(train_vars_dims):
            " Fully Connected Layers "
            train_vars = []
            current_y_hat = self.x_frames
            for j in range(fully_connected_layers):
                # layer n + m: fully connected
                W, b, z_hat, y_hat = layers.fully_connected(
                    name, "full_"+str(j + 1)+"_"+str(i), current_y_hat, dim_in[i][j], dim_out[i][j],
                    tf.random_normal_initializer(stddev=1.0 / np.sqrt(dim_in[i][j]), seed=SEED), gate_fun)

                current_y_hat = y_hat
                train_vars.extend([W, b])
            y_hats.append(current_y_hat)
            self.train_vars.append(train_vars)

        combined_y_hat = tf.concat(y_hats, 1)

        """ Output layer """
        # output layer: fully connected
        if reward_path:
            final_dim_in = dim_out[0][-1] + dim_out[1][-1]
        else:
            final_dim_in = dim_out[0][-1]
        W, b, z_hat, self.y_hat = layers.fully_connected(
            name, "output_layer", combined_y_hat, final_dim_in, actions,
            tf.random_normal_initializer(stddev=1.0 / np.sqrt(final_dim_in), seed=SEED), linear_transfer)
        for lst in self.train_vars:
            lst.extend([W, b])

        # Obtaining y_hat and Scaling by the Importance Sampling
        y_hat = tf.gather_nd(self.y_hat, self.x_actions)
        y_hat = tf.multiply(y_hat, self.isampling)
        y = tf.multiply(self.y, self.isampling)
        # Temporal Difference Error
        self.td_error = tf.subtract(y_hat, y)
        # Loss
        self.train_loss = tf.reduce_sum(tf.pow(self.td_error, 2)) # Squared TD error
########################################################################################################################
########################################################################################################################
