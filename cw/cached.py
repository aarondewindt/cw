""""These are function decorators for cached properties. The property's getter is only called once when it's first
requested. Subsequent calls return the cached value from that first call."""

# Source: http://code.activestate.com/recipes/276643-caching-and-aliasing-with-descriptors/

import inspect

__maintainer__ = "Aaron M. de Windt"


# I know these class names don't follow PEP-8, but they are technically function decorators.
class cached(object):
    """
    Computes attribute value and caches it in instance. Use ``del inst.myMethod`` to clear cache.
    """

    def __init__(self, method, name=None):
        self.method = method
        self.name = name
        self.__doc__ = inspect.getdoc(self.method)
        if 'return' in method.__annotations__:
            self.__annotations__ = { 'return': method.__annotations__['return']}

    def __set_name__(self, owner, name):
        self.name = self.name or name

    def __get__(self, inst, cls):
        if inst is None:
            return self
        result = self.method(inst)
        setattr(inst, self.name, result)
        return result


class cached_class(object):
    """
    Computes attribute value and caches it in class.  Use ``del MyClass.myMethod`` to clear cache.
    """

    def __init__(self, method, name=None):
        self.method = method
        self.name = name or method.__name__
        self.__doc__ = inspect.getdoc(self.method)
        self.__annotations__ = method.__annotations__

    def __get__(self, inst, cls):
        result = self.method(cls)
        setattr(cls, self.name, result)
        return result
