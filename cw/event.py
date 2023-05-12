import asyncio
from collections.abc import Iterable


class Event:
    """
    Event objects are used to trigger events. Event handling functions can be added to
    it and are then called when the event object is called.

    .. note:: Asyncio co-routine are supported.

    :param asyncio.BaseEventLoop loop: Default event loop on which to run coroutines.
       If ``None``, coroutines cannot be called.
    """
    def __init__(self, loop=None):
        self.handlers = []
        self.default_loop = loop

    def __iadd__(self, handler):
        """
        Adds one or more handlers to the event.

        :param handler: A callable or list of callables to add to the event.
        """

        # Check if an iterable was passed.
        if isinstance(handler, Iterable):
            # Add each element in the iterable.
            for h in handler:
                self.__add_handler(h)
        else:
            # Add the handler
            self.__add_handler(handler)

        return self

    def __add_handler(self, handler):
        """
        Adds a single callable to the event.

        :param handler: Callable object that will handle the event.
        """
        # Check if handler is a callable and not added yet to the event handlers list.
        if callable(handler):
            if handler not in self.handlers:
                self.handlers.append(handler)
        else:
            raise TypeError("Handler must be callable.")

    def __isub__(self, handler):
        """
        Removes one or more handlers from the event.

        :param handler: Callable or list of callable to remove.
        """
        if handler in self.handlers:
            self.handlers.remove(handler)

        return self

    def __call__(self, *args, **kwargs):
        """
        Raises the event. This will result in all handlers being called. All parameters will be passed to all handlers.

        :param args:
        :param kwargs:
        """

        for handler in self.handlers:
            if asyncio.iscoroutinefunction(handler):
                if self.default_loop is None:
                    raise TypeError("No default asyncio event loop set for this event, so coroutines are not supported.")
                asyncio.run_coroutine_threadsafe(handler(*args, **kwargs), self.default_loop)
            else:
                handler(*args, **kwargs)
