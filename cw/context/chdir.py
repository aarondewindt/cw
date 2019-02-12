from contextlib import contextmanager
import os


@contextmanager
def chdir(path):
    """
    Context manager that temporarily changes the current working directory.

    :param path: Path to the target current working directory.
    """
    # Store the current working directory path and change tot the new one
    old_path = os.getcwd()
    os.chdir(str(path))

    # Yield to allow managed code to run.
    yield

    # Change directory back.
    os.chdir(old_path)
