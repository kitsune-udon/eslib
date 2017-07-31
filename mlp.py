import eslib
import eslib.links as L
import eslib.functions as F

class MLP(eslib.Chain):
    def __init__(self, n_actions):
        super(MLP, self).__init__()
        n_hiddens = 20
        with self.init_scope():
            self.l1 = L.Linear(None, n_hiddens)
            self.l2 = L.Linear(None, n_hiddens)
            self.l3 = L.Linear(None, n_hiddens)
            self.l4 = L.Linear(None, n_actions)
    def __call__(self, x):
        h = self.l1(x)
        h = self.l2(h)
        h = self.l3(h)
        h = self.l4(h)
        return h
