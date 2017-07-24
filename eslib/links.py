from eslib import link
from eslib import parameter
from eslib.initializers import Constant
import eslib.functions as F
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
    def __init__(self, in_channel, out_channel, W_sigma=0.1, b_sigma=0.1, nobias=False):
        super(Linear, self).__init__()
        assert(out_channel is not None)
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.nobias = nobias
        with self.init_scope():
            self.W = parameter.Parameter(initializer=Constant(0.), sigma=W_sigma)
            if not nobias:
                self.b = parameter.Parameter(initializer=Constant(0.), sigma=b_sigma)
        if in_channel is not None:
            W_shape = (out_channel, in_channel)
            self.W.initialize(W_shape)
        if not nobias:
            b_shape = out_channel
            self.b.initialize(b_shape)

    def __call__(self, x):
        if self.W.data is None:
            if self.in_channel is None:
                self.in_channel = x.shape[0]
            W_shape = (self.out_channel, self.in_channel)
            self.W.initialize(W_shape)
        r = np.dot(self.W(), x)
        if not self.nobias:
            r += self.b()
        return r

class LSTMBase(link.Chain):
    def __init__(self, in_size, out_size):
        super(LSTMBase, self).__init__()
        self.state_size = out_size

        with self.init_scope():
            self.upward = Linear(in_size, 4 * out_size)
            self.lateral = Linear(out_size, 4 * out_size, nobias=True)
            if in_size is not None:
                self._initialze_params()

    def _initialze_params(self):
        a, i, f, o = F._extract_gates(
                self.upward.b.data.reshape(4 * self.state_size, 1))
        Constant(0.)(a)
        Constant(0.)(i)
        Constant(1.)(f)
        Constant(0.)(o)

class LSTM(LSTMBase):
    def __init__(self, in_size, out_size):
        super(LSTM, self).__init__(in_size, out_size)
        self.reset_state()

    def reset_state(self):
        self.c = self.h = None

    def __call__(self, x):
        lstm_in = self.upward(x)
        if self.h is not None:
            lstm_in += self.lateral(self.h)
        if self.c is None:
            self.c = np.zeros(self.state_size, dtype=x.dtype)
        self.c, y = F.lstm(self.c, lstm_in)
        self.h = y
        return y

