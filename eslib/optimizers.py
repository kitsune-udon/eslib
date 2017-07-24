from eslib import optimizer
import numpy as np
import math

_default_hyperparam_sgd = optimizer.Hyperparameter()
_default_hyperparam_sgd.lr = 0.01

class SGDRule(optimizer.UpdateRule):
    def __init__(self, parent_hyperparam=None, lr=None):
        super(SGDRule, self).__init__(parent_hyperparam or _default_hyperparam_sgd)
        if lr is not None:
            self.hyperparam.lr = lr
    def update_core(self, param):
        grad = param.grad
        if grad is None:
            raise RuntimeError("grad is unallocated")
        param.data += self.hyperparam.lr * grad

class SGD(optimizer.GradientMethod):
    def __init__(self, lr=_default_hyperparam_sgd.lr):
        super(SGD, self).__init__()
        self.hyperparam.lr = lr
    def create_update_rule(self):
        return SGDRule(self.hyperparam)

_default_hyperparam_adam = optimizer.Hyperparameter()
_default_hyperparam_adam.alpha = 0.001
_default_hyperparam_adam.beta1 = 0.9
_default_hyperparam_adam.beta2 = 0.999
_default_hyperparam_adam.eps = 1e-8

class AdamRule(optimizer.UpdateRule):
    def __init__(self, parent_hyperparam=None,
            alpha=None, beta1=None, beta2=None, eps=None):
        super(AdamRule, self).__init__(
                parent_hyperparam or _default_hyperparam_adam)
        if alpha is not None:
            self.hyperparam.alpha = alpha
        if beta1 is not None:
            self.hyperparam.beta1 = beta1
        if beta2 is not None:
            self.hyperparam.beta2 = beta2
        if eps is not None:
            self.hyperparam.eps = eps

    def init_state(self, param):
        self.state['m'] = np.zeros_like(param.data)
        self.state['v'] = np.zeros_like(param.data)

    def update_core(self, param):
        grad = param.grad
        if grad is None:
            return
        hp = self.hyperparam
        m, v = self.state['m'], self.state['v']
        m += (1 - hp.beta1) * (grad - m)
        v += (1 - hp.beta2) * (grad * grad - v)
        param.data += self.lr * m / (np.sqrt(v) + hp.eps)

    @property
    def lr(self):
        fix1 = 1. - math.pow(self.hyperparam.beta1, self.t)
        fix2 = 1. - math.pow(self.hyperparam.beta2, self.t)
        return self.hyperparam.alpha * math.sqrt(fix2) / fix1

class Adam(optimizer.GradientMethod):
    def __init__(self,
            alpha=_default_hyperparam_adam.alpha,
            beta1=_default_hyperparam_adam.beta1,
            beta2=_default_hyperparam_adam.beta2,
            eps=_default_hyperparam_adam.eps):
        super(Adam, self).__init__()
        self.hyperparam.alpha = alpha
        self.hyperparam.beta1 = beta1
        self.hyperparam.beta2 = beta2
        self.hyperparam.eps = eps

    def create_update_rule(self):
        return AdamRule(self.hyperparam)

    @property
    def lr(self):
        fix1 = 1. - math.pow(self.hyperparam.beta1, self.t)
        fix2 = 1. - math.pow(self.hyperparam.beta2, self.t)
        return self.hyperparam.alpha * math.sqrt(fix2) / fix1

_default_hyperparam_smorm3 = optimizer.Hyperparameter()
_default_hyperparam_smorm3.lr = 0.001
_default_hyperparam_smorm3.eps = 1e-16

class SMORMS3Rule(optimizer.UpdateRule):
    def __init__(self, parent_hyperparam=None, lr=None, eps=None):
        super(SMORMS3Rule, self).__init__(
                parent_hyperparam or _default_hyperparam_smorm3)
        if lr is not None:
            self.hyperparam.lr = lr
        if eps is not None:
            self.hyperparam.eps = eps

    def init_state(self, param):
        self.state['mem'] = np.ones_like(param.data)
        self.state['g'] = np.zeros_like(param.data)
        self.state['g2'] = np.zeros_like(param.data)

    def update_core(self, param):
        grad = param.grad
        if grad is None:
            return

        mem, g, g2 = self.state['mem'], self.state['g'], self.state['g2']
        r = 1 / (mem + 1)
        g = (1 - r) * g + r * grad
        g2 = (1 - r) * g2 + r * grad * grad
        x = g * g / (g2 + self.hyperparam.eps)
        param.data += grad * np.minimum(x, self.hyperparam.lr) \
                / (np.sqrt(g2) + self.hyperparam.eps)
        mem = 1 + mem * (1 - x)

        self.state['mem'], self.state['g'], self.state['g2'] = mem, g, g2

class SMORMS3(optimizer.GradientMethod):
    def __init__(self, lr=_default_hyperparam_smorm3.lr, eps=_default_hyperparam_smorm3.eps):
        super(SMORMS3, self).__init__()
        self.hyperparam.lr = lr
        self.hyperparam.eps = eps

    def create_update_rule(self):
        return SMORMS3Rule(self.hyperparam)
