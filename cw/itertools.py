from itertools import zip_longest
from typing import Iterable, Any, TypeVar, Sequence
from cw.singletons import Singleton
from functools import partial


T = TypeVar('T')

# def zip_


class EndItemType(Singleton):
    name = "EndItem"


EndItem = EndItemType()


def grouper(iterable: Iterable[T], n: int, fillvalue: Any=None) -> Iterable[Sequence[T]]:
    """Collect data into fixed-length chunks or blocks

    :param iterable:
    :param int n: Group size
    :param fillvalue: Value used to pad out the last group.
    """
    # Copied from the python docs: https://docs.python.org/3/library/itertools.html
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    yield from zip_longest(*args, fillvalue=fillvalue)


def chunks(iterable: Iterable[T], n: int) -> Iterable[Sequence[T]]:
    """
    Collect data into chunks with a maximum length.

    :param iterable:
    :param int n: Maximum chunk size.
    """
    for g in grouper(iterable, n, EndItem):
        yield tuple(until(g, EndItem))


def until(g: Iterable[T], value: Any) -> Iterable[T]:
    """
    Yield until the value is found in g.

    :param g: Iterable
    :param value: Stop value
    """
    for x in g:
        if x == value:
            break
        yield x