import functools
from .module_base import ModuleBase


def alias_states(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        module: ModuleBase = args[0]
        module.s = module.simulation.states
        result = f(*args, **kwargs)
        del module.s
        return result
    return wrapper
