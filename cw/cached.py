""""These are function decorators for cached properties. The property's getter is only called once when it's first
requested. Subsequent calls return the cached value from that first call."""

# Source: http://code.activestate.com/recipes/276643-caching-and-aliasing-with-descriptors/
from typing import TypeVar, Callable, Literal, overload, Union

import inspect

__maintainer__ = "Aaron M. de Windt"


# I know these class names don't follow PEP-8, but they are technically function decorators.
class cached(object):
    """
    Computes attribute value and caches it in instance. Use ``del inst.myMethod`` to clear cache.
    """

    def __init__(self, method, name=None):
        self.method = method
        self.setter_method = None
        self.name = name
        self.__doc__ = inspect.getdoc(self.method)

    def setter(self, method):
        self.setter_method = method
        return self

    def __set_name__(self, owner, name):
        self.name = self.name or name
        self.cache_name = f"__{name}_cached_value"

    def __get__(self, instance, cls):
        if instance is None:
            return self

        if hasattr(instance, self.cache_name):
            return getattr(instance, self.cache_name)

        result = self.method(instance)
        setattr(instance, self.cache_name, result)
        return result

    def __set__(self, instance, value):
        if instance is None:
            return

        if self.setter_method:
            self.setter_method(instance, value)
            setattr(instance, self.cache_name, value)
        else:
            raise AttributeError("can't set attribute")

    def __delete__(self, instance):
        if hasattr(instance, self.cache_name):
            delattr(instance, self.cache_name)


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
