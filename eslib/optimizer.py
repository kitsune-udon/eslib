import collections
import numpy as np

class Hyperparameter:
    def __init__(self, parent=None):
        self._parent = parent

    def __getattr__(self, name):
        if '_parent' not in self.__dict__:
            raise AttributeError
        return getattr(self._parent, name)

class UpdateRule:
    def __init__(self, parent_hyperparam=None):
        self._hooks = collections.OrderedDict()
        self._state = None
        self.enabled = True
        self.hyperparam = Hyperparameter(parent_hyperparam)
        self.t = 0

    @property
    def state(self):
        return self._state

    def add_hook(self, hook):
        if not callable(hook):
            raise TypeError

        name = getattr(hook, 'name', None)

        if name is None:
            raise ValueError

        if name in self._hooks:
            raise ValueError

        self._hooks[name] = hook

    def remove_hook(self, name):
        del self._hooks[name]

    def update(self, param):
        if not self.enabled:
            return

        self.t += 1
        self._prepare(param)

        for hook in self._hooks.values():
            hook(self, param)

        self.update_core(param)

    def update_core(self, param):
        raise NotImplementedError

    def init_state(self):
        pass

    def _prepare(self, param):
        state = self.state
        if state is None:
            self._state = {}
            self.init_state(param)

class Optimizer:
    def setup(self, link):
        self.target = link
        self.t = 0
        self._hooks = collections.OrderedDict()

    def update(self):
        raise NotImplementedError

    def add_hook(self, hook):
        if not callable(hook):
            raise TypeError

        if not hasattr(self, '_hooks'):
            raise RuntimeError

        name = hook.name
        if name in self._hooks:
            raise KeyError

        self._hooks[name] = hook

    def remove_hook(self, name):
        del self._hooks[name]

    def call_hooks(self):
        for hook in self._hooks.values():
            self._call_hook(hook)

    def _call_hook(self, hook):
        if getattr(hook, 'call_for_each_param', False):
            for param in self.target.params(False):
                hook(param.update_rule, param)
        else:
            hook(self)

class GradientMethod(Optimizer):
    def __init__(self):
        super(GradientMethod, self).__init__()
        self.hyperparam = Hyperparameter()
    def setup(self, link):
        super(GradientMethod, self).setup(link)
        for param in link.params():
            param.update_rule = self.create_update_rule()
    def create_update_rule(self):
        raise NotImplementedError
    def update(self):
        self.call_hooks()
        self.t += 1
        for param in self.target.params(False):
            param.update()

def _sum_sqnorm(arr):
    acc = 0
    for x in arr:
        x = x.ravel()
        s = x.dot(x)
        acc += s
    return acc

class GradientClipping:
    name = "GradientClipping"
    def __init__(self, threshold, eps=1e-8):
        self.threshold = threshold
        self.eps = eps
    def __call__(self, opt):
        norm = np.sqrt(_sum_sqnorm(
            [p.grad for p in opt.target.params(False)])) + self.eps
        rate = self.threshold / norm
        if rate < 1:
            for param in opt.target.params(False):
                grad = param.grad
                grad *= rate

class GradientHardClipping:
    name = "GradientHardClipping"
    call_for_each_param = True
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
    def __call__(self, rule, param):
        grad = param.grad
        np.clip(grad, self.lower_bound, self.upper_bound, out=grad)

class WeightDecay:
    name = "WeightDecay"
    call_for_each_param = True
    def __init__(self, rate):
        self.rate = rate
    def __call__(self, rule, param):
        p, g = param.data, param.grad
        g -= self.rate * p
