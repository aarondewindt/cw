import cProfile
import contextlib
import pstats
from io import StringIO


@contextlib.contextmanager
def profile_it(*args, run=True, **kwargs):
    """
    Context manager that profiles the code inside.
    :param args:
    :param run: True to profile the code, otherwise the code will run as normal.
    :param kwargs:
    :return:
    """
    if run:
        profile = cProfile.Profile(*args, **kwargs)

        profile.enable()
        yield
        profile.disable()

        s = StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(profile, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
    else:
        yield
