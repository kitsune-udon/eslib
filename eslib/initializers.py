from eslib import initializer
import numpy as np

class Constant(initializer.Initializer):
    def __init__(self, fill_value):
        self.fill_value = fill_value
        super(Constant, self).__init__()
    def __call__(self, array):
        array[...] =  np.asarray(self.fill_value)
