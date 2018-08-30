import numpy as np


class Buffer:
    def __init__(self, buffer_size, observation_dimensions):
        self.buffer_size = buffer_size
        self.dim = [buffer_size]
        self.dim.extend(observation_dimensions)
        self.buffer_frames = np.zeros(shape=self.dim, dtype=float)
        self.buffer_actions = np.zeros(shape=[self.buffer_size, 1], dtype=int)
        self.buffer_labels = np.zeros(shape=[self.buffer_size], dtype=float)
        self.current_buffer_size = 0
        self.buffer_full = False

    def add_to_buffer(self, buffer_entry):
        if self.current_buffer_size == self.buffer_size:
            self.reset()
        self.buffer_frames[self.current_buffer_size] = buffer_entry[0]
        self.buffer_actions[self.current_buffer_size] = buffer_entry[1]
        self.buffer_labels[self.current_buffer_size] = buffer_entry[2]
        self.current_buffer_size += 1
        if self.current_buffer_size == self.buffer_size:
            self.buffer_full = True

    def sample(self, batch_size):
        if self.current_buffer_size < batch_size:
            raise ValueError("Not enough entries in the buffer.")
        sample_frames = self.buffer_frames[:batch_size]
        sample_actions = self.buffer_actions[:batch_size]
        sample_labels = self.buffer_labels[:batch_size]
        return sample_frames, sample_actions, sample_labels

    def reset(self):
        self.buffer_frames = np.zeros(shape=self.dim, dtype=float)
        self.buffer_actions = np.zeros(shape=[self.buffer_size, 1], dtype=int)
        self.buffer_labels = np.zeros(shape=[self.buffer_size], dtype=float)
        self.current_buffer_size = 0
        self.buffer_full = False
