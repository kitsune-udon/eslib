import numpy as np
import eslib

class Parameter:
    def __init__(self, name=None, shape=None, initializer=None, sigma=0.1):
        self.name = name
        self.shape = shape
        self.initializer = initializer
        self.data = None
        self.grad = None
        self.ptb = None
        self.sigma = sigma
        self.update_rule = None
    def cleargrad(self):
        self.grad = None
    def update(self):
        if self.update_rule is not None:
            self.update_rule.update(self)
    def __call__(self):
        if self.data is None:
            raise RuntimeError("data is unallocated")
        if eslib.within_perturbation_scope():
            if self.ptb is None:
                raise RuntimeError("ptb is unallocated")
            return self.data + self.ptb
        else:
            return self.data
    def initialize(self, shape):
        if shape is None:
            raise RuntimeError("invalid argument")
        if self.initialize is None:
            raise RuntimeError("initialzier is unallocated")
        self.shape = shape
        array = np.empty(shape, dtype=eslib.dtype)
        self.initializer(array)
        self.data = array

