import sys
import traceback
import asyncio


def log_exc(log, catch=False):
    """
    Decorator that logs uncaught exceptions in a function.

    :param log: :class:`logging.Logger` object to log the exception to.
    :param catch: True to catch the exception and prevent it from propagating.
    """

    # This function (log_exc) is not the decorator itself, instead it's the
    # factory for the decorator. The reason we do this is because we want to
    # pass extra parameters to the decorator and this is the only option.

    # Create the decorator function
    def decorator(target):
        # Create the appropriate wrapper function.
        # If the target is a coroutine,
        # so should the wrapper be.
        if asyncio.iscoroutinefunction(target):
            async def wrapper(*args, **kwargs):
                try:
                    return await target(*args, **kwargs)
                except:
                    # Log the exception, and if it should be not caught re-raise is.
                    log.exception(sys.exc_info())
                    if not catch:
                        raise
        else:
            def wrapper(*args, **kwargs):
                try:
                    return target(*args, **kwargs)
                except:
                    # Log the exception, and if it should be not caught re-raise is.
                    log.exception(sys.exc_info())
                    if not catch:
                        raise

        # Make sure the decorated function has the same name, documentation
        # and string representation as the original function.
        wrapper.__name__ = target.__name__
        wrapper.__doc__ = target.__doc__
        wrapper.__repr__ = target.__repr__
        return wrapper
    return decorator


# Create the decorator function
def print_exc(target):
    # Create the appropriate wrapper function.
    # If the target is a coroutine,
    # so should the wrapper be.
    if asyncio.iscoroutinefunction(target):
        async def wrapper(*args, **kwargs):
            try:
                return await target(*args, **kwargs)
            except Exception:
                # Print the exception, and if it should be not caught re-raise is.
                traceback.print_exception(*sys.exc_info())
                raise
    else:
        def wrapper(*args, **kwargs):
            try:
                return target(*args, **kwargs)
            except Exception:
                # Print the exception, and if it should be not caught re-raise is.
                traceback.print_exception(*sys.exc_info())
                raise

    # Make sure the decorated function has the same name, documentation
    # and string representation as the original function.
    wrapper.__name__ = target.__name__
    wrapper.__doc__ = target.__doc__
    wrapper.__repr__ = target.__repr__
    return wrapper
