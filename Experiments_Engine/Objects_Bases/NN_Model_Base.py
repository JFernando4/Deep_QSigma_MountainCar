import abc
import tensorflow as tf

class ModelBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, model_dictionary=None):
        self._model_dictionary = model_dictionary

    @abc.abstractmethod
    def get_model_dictionary(self):
        return self._model_dictionary

    @abc.abstractmethod
    def replace_model_weights(self, new_vars, tf_session=tf.Session()):
        pass

    @abc.abstractmethod
    def get_variables(self, tf_session=tf.Session()):
        pass

    @staticmethod
    @abc.abstractmethod
    def print_number_of_parameters(parameter_list):
        sess = tf.Session()
        total = 0
        for layer in range(int(len(parameter_list)/2)):
            index = layer * 2
            layer_total = sess.run(tf.size(parameter_list[index]) + tf.size(parameter_list[index+1]))
            print("Number of parameters in layer", layer + 1, ":", layer_total)
            total += layer_total
        print("Total number of parameters:", total)

    @abc.abstractmethod
    def check_model_dictionary(self):
        pass
