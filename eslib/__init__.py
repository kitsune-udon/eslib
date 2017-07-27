import numpy as np
from eslib.perturbation import Perturbation
from eslib.link import Link, Chain
import contextlib,sys


dtype = np.float32

def within_perturbation_scope():
    return getattr(sys.modules[__name__], '_within_perturbation_scope', False)

@contextlib.contextmanager
def perturbation_scope():
    obj = sys.modules[__name__]
    old_flag = obj.within_perturbation_scope()
    obj._within_perturbation_scope = True
    try:
        yield
    finally:
        obj._within_perturbation_scope = old_flag

def _alloc_ptb(param, ptb):
    param.ptb = param.sigma * ptb.generate(param.shape)

def _acc_ptb_score_mul(param, score):
    if not np.isscalar(score):
        raise TypeError("score is not scalar type")
    if param.grad is None:
        param.grad = np.zeros_like(param.data)
    param.grad += param.ptb * score

def set_perturbations(link, ptb, ptb_id):
    with ptb.generation_scope(ptb_id):
        for param in link.params(False):
            _alloc_ptb(param, ptb)

def calculate_grads(link, ptb, scores):
    n = len(scores)
    for ptb_id in range(len(scores)):
        set_perturbations(link, ptb, ptb_id)
        for param in link.params(False):
            _acc_ptb_score_mul(param, scores[ptb_id])
    for param in link.params(False):
        c = 1. / (n * param.sigma)
        param.grad *= c

def fix_model(model, x):
    model(x)

def fitness_shaping(scores, cutoff_flag):
    n = len(scores)
    rank = scores.argsort()[::-1].argsort() + 1
    t = np.log(n / 2 + 1) - np.log(rank)
    if cutoff_flag:
        t[t < 0] = 0
    u = t / t.sum() - 1 / n
    return u
