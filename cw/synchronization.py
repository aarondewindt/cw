from threading import Condition
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





