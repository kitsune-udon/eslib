import collections
import contextlib
from eslib import parameter

class Link:
    def __init__(self):
        self._params = collections.OrderedDict()
        self.name = None

    @property
    def within_init_scope(self):
        return getattr(self, '_within_init_scope', False)

    @contextlib.contextmanager
    def init_scope(self):
        old_flag = self.within_init_scope
        self._within_init_scope = True
        try:
            yield
        finally:
            self._within_init_scope = old_flag

    def __setattr__(self, name, value):
        if self.within_init_scope and isinstance(value, parameter.Parameter):
            value.name = name
            self._params[name] = name
        super(Link, self).__setattr__(name, value)

    def __delattr__(self, name):
        del self._params[name]
        super(Link, self).__delattr__(name, value)

    def params(self, include_uninit=True):
        d = self.__dict__
        for name in self._params:
            if include_uninit or d[name].data is not None:
                yield d[name]

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()

    def enable_update(self):
        for param in self.params():
            rule = param.update_rule
            if rule is not None:
                rule.enabled = True

    def disable_update(self):
        for param in self.params():
            rule = param.update_rule
            if rule is not None:
                rule.enabled = False

    @property
    def update_enabled(self):
        for param in self.params():
            rule = param.update_rule
            if rule is not None and rule.enabled:
                return True
        return False

class Chain(Link):
    def __init__(self):
        super(Chain, self).__init__()
        self._children = collections.OrderedDict()

    def __setattr__(self, name, value):
        if self.within_init_scope and isinstance(value, Link):
            if hasattr(self, name):
                raise AttributeError
            value.name = name
            self._children[name] = name
        super(Chain, self).__setattr__(name, value)

    def __delattr__(self, name):
        del self._children[name]
        super(Chain, self).__delattr__(name)

    def params(self, include_uninit=True):
        for param in super(Chain, self).params(include_uninit):
            yield param
        d = self.__dict__
        for name in self._children:
            for param in d[name].params(include_uninit):
                yield param
