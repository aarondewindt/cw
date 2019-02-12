import asyncio

# To my future maintainers. I'm sorry for this function in particular, but
# I didn't have any other option. This thing makes unit-testing coroutines
# so much easier. Imagine having to write half of this in each unit-test.


# TODO: Write unittest for async_test.

def async_test(loop=None, timeout=None):
    """
    Decorator enabling co-routines to be run in python unittests.

    :param loop: Event loop in which to run the co-routine. By default its ``asyncio.get_event_loop()``.
    :param timeout: Test timeout in seconds.
    """
    loop = loop or asyncio.get_event_loop()

    def async_test_decorator(f):
        def wrapper(*args, **kwargs):
            if loop is asyncio.get_event_loop():
                loop.run_until_complete(f(*args, **kwargs))
            else:
                future = asyncio.run_coroutine_threadsafe(f(*args, **kwargs), loop=loop)
                future.result(timeout=timeout)

        # Make sure the decorated function has the same name, documentation
        # and string representation as the original function.
        wrapper.__name__ = f.__name__
        wrapper.__doc__ = f.__doc__
        wrapper.__repr__ = f.__repr__
        return wrapper
    return async_test_decorator


