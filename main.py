import eslib
import eslib.links as L
import eslib.optimizers as O
from eslib.optimizer import GradientClipping, WeightDecay
import numpy as np

class MLP(eslib.Chain):
    def __init__(self):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(1, 1)
    def __call__(self, x):
        h = self.l1(x)
        return h

def sample():
    r = np.random.uniform(-1, 1, size=1)
    return r

def get_score(model):
    x = sample()
    y = model(x)
    t = y[0]-1
    return -t*t

n_iters = 200
n_ptbs = 2
p = eslib.Perturbation()
model = MLP()
optimzier = O.SMORMS3(lr=1e-2)
optimzier.setup(model)
optimzier.add_hook(GradientClipping(1e2))
optimzier.add_hook(WeightDecay(0.005))
eslib.fix_model(model, np.array([1.]))

for i in range(n_iters):
    scores = []
    with eslib.perturbation_scope():
        for ptb_id in range(n_ptbs):
            eslib.set_perturbations(model, p, ptb_id)
            score = get_score(model)
            scores.append(score)
    eslib.calculate_grads(model, p, scores)
    optimzier.update()
    model.cleargrads()
    p.age()

print(get_score(model))
print(model.l1.W.data)
print(model.l1.b.data)
