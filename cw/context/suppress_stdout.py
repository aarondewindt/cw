from contextlib import contextmanager
import sys


@contextmanager
def suppress_stdout(suppress=True):
    if suppress:
        old_stdout = sys.stdout
        sys.stdout = write_null
        yield
        sys.stdout = old_stdout
    else:
        yield


class WriteNull:
    def write(self, *args, **kwargs):
        pass

    flush = write


write_null = WriteNull()