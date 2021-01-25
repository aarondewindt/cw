from threading import Condition, Lock
from time import monotonic as _time
import os


class CheckInSemaphore:
    """
    Thread synchronization object used to keep a thread waiting until
    all checked in threads have checked out.
    """

    def __init__(self):
        self.tokens = []  #: List of checked in tokens.
        self.condition = Condition()
        """
        Condition variable used internally.
        """

    def wait(self, timeout=None):
        """
        Wait for all checked in threads to check out.

        :param timeout: Time out.
        """
        with self.condition:
            self.condition.wait_for(lambda: len(self.tokens) == 0, timeout=timeout)

    def check_in(self, token=None):
        """
        Used to check in a thread.

        :param token: Optional unique token used to identify
           the thread checking in. This token must be unique,
           duplicate tokens will be ignored. If no token is given,
           one will be generated.
        :returns: The token.
        """

        if token is None:
            # No toke was passed, so we must generate one.
            token = os.urandom(32)

            # If the generated token is already in the token list.
            # generate a new one.
            while token in self.tokens:
                token = os.urandom(32)

        # Add the token if it's not in the list.
        if token not in self.tokens:
            self.tokens.append(token)

        return token

    def check_out(self, token):
        """
        Used to check out a thread.

        :param token: The token that was used to check in.
        """
        if token in self.tokens:
            self.tokens.remove(token)

        with self.condition:
            self.condition.notify_all()


class BinarySemaphore:
    """This class implements binary semaphore objects.

    Semaphores manage a counter representing the number of release() calls minus
    the number of acquire() calls, plus an initial value. The acquire() method
    blocks if necessary until it can return without making the counter
    negative. If not given, value defaults to 1.

    """

    # After Tim Peters' semaphore class, but not quite the same (no maximum)

    def __init__(self, initial_value: bool=True):
        self._cond = Condition(Lock())
        self._value = initial_value

    def acquire(self, blocking=True, timeout=None):
        """Acquire a semaphore, decrementing the internal counter by one.

        When invoked without arguments: if the internal counter is larger than
        zero on entry, decrement it by one and return immediately. If it is zero
        on entry, block, waiting until some other thread has called release() to
        make it larger than zero. This is done with proper interlocking so that
        if multiple acquire() calls are blocked, release() will wake exactly one
        of them up. The implementation may pick one at random, so the order in
        which blocked threads are awakened should not be relied on. There is no
        return value in this case.

        When invoked with blocking set to true, do the same thing as when called
        without arguments, and return true.

        When invoked with blocking set to false, do not block. If a call without
        an argument would block, return false immediately; otherwise, do the
        same thing as when called without arguments, and return true.

        When invoked with a timeout other than None, it will block for at
        most timeout seconds.  If acquire does not complete successfully in
        that interval, return false.  Return true otherwise.

        """
        if not blocking and timeout is not None:
            raise ValueError("can't specify timeout for non-blocking acquire")

        with self._cond:
            if self._value:
                self._value = False
                return True
            else:
                if not blocking:
                    return False
                else:
                    if self._cond.wait(timeout):
                        self._value = False
                        return True
                    else:
                        return False

    __enter__ = acquire

    def release(self):
        """Release a semaphore, incrementing the internal counter by one.

        When the counter is zero on entry and another thread is waiting for it
        to become larger than zero again, wake up that thread.

        """
        if not self._value:
            with self._cond:
                self._value = True
                self._cond.notify()

    def __exit__(self, t, v, tb):
        self.release()


