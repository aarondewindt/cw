from contextlib import contextmanager
from time import perf_counter


@contextmanager
def time_it(name, run=True):
    """
    Times the code withing the context manager.

    :param name: Name of the timer.
    :param run: True to time the code, otherwise the code will as normal.
    :return:
    """
    if run:
        t0 = perf_counter()
        yield
        print(f"{name}: {perf_counter() - t0} [s]")
    else:
        yield
