from eslib import link
from eslib import parameter
from eslib.initializers import Constant
import numpy as np

class Bias(link.Link):
    def __init__(self, n_units, b_sigma=0.1):
        super(Bias, self).__init__()
        assert(n_units is not None)
        with self.init_scope():
            self.b = parameter.Parameter(initializer=Constant(0.), sigma=b_sigma)
        self.b.initialize(n_units)
    def __call__(self, x):
        r = x + self.b()
        return r

class Linear(link.Link):
    def __init__(self, in_channel, out_channel, W_sigma=0.1, b_sigma=0.1):
        super(Linear, self).__init__()
        assert(out_channel is not None)
        self.in_channel = in_channel
        self.out_channel = out_channel
        with self.init_scope():
            self.W = parameter.Parameter(initializer=Constant(0.), sigma=W_sigma)
            self.b = parameter.Parameter(initializer=Constant(0.), sigma=b_sigma)
        if in_channel is not None:
            W_shape = (out_channel, in_channel)
            self.W.initialize(W_shape)
        b_shape = out_channel
        self.b.initialize(b_shape)

    def __call__(self, x):
        if self.W.data is None:
            if self.in_channel is None:
                self.in_channel = x.shape[0]
            W_shape = (self.out_channel, self.in_channel)
            self.W.initialize(W_shape)
        r = np.dot(self.W(), x) + self.b()
        return r
