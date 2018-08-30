"""
MIT License

Copyright (c) 2017 Amii

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np


class CircularBuffer:
    def __init__(self, maxlen, shape, dtype):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = np.empty((maxlen,) + shape, dtype=dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if idx < 0 or idx >= self.length:
                raise KeyError()
        elif isinstance(idx, np.ndarray):
            if (idx < 0).any() or (idx >= self.length).any():
                raise KeyError()
        elif isinstance(idx, slice):
            if (idx.start < 0) or (idx.start >= self.length) or (idx.stop < 0) or (idx.stop > self.length):
                raise KeyError
            else:
                return self.data.take(np.arange(self.start + idx.start, self.start + idx.stop), axis=0, mode='wrap')
        return self.data.take(self.start + idx, mode='wrap', axis=0)

    def __array__(self):
        return self.data.take(np.arange(self.start, self.start + self.length), mode='wrap', axis=0)

    def append(self, v):
        if self.length < self.maxlen:
            self.length += 1
        elif self.length == self.maxlen:
            self.start = (self.start + 1) % self.maxlen
        else:
            raise RuntimeError()

        self.data[(self.start + self.length - 1) % self.maxlen] = v

    def take(self, idx):
        return self.data.take(self.start + idx, mode='wrap', axis=0)
