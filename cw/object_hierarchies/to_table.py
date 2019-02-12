from pathlib import PurePosixPath
from collections import OrderedDict, defaultdict
import numpy as np
from itertools import product


from typing import Dict, Union, Iterable, Tuple, Any


def object_hierarchy_to_tables(obj) -> Dict[int, Dict]:
    """
    Transforms a hierarchy of dictionaries and lists into a structure that allows the
    data to be saved in 2d table formats such as csv and excel.

    .. note: End point elements may not be of type list. Currently only list/matrices
             of numbers are supported with the use of numpy arrays. This will change in the fu

    :param obj: Hierarchy object to be transformed.
    :return: A dictionary with integers as keys. The value of the key is the length
             of element. A length of 0 indicates that it's a scalar element, while a
             a length of 1 means its a list with one element.
    """

    # Initialize the table to a default dictionary with dictionary elements.
    tables = defaultdict(dict)

    path = PurePosixPath("")

    # Walk through all end points in the object hierarchy.
    for obj, path in walker(obj, path):
        # If it's a scalar elements, add it to the 0 length table.
        if np.isscalar(obj):
            tables[0][str(path).replace("/", ".")] = obj
        else:
            try:
                # Try to get the length and add it to that table.
                obj_len = len(obj)
            except TypeError:
                # If it was not possible add it as a scalar element.
                tables[0][str(path).replace("/", ".")] = obj
            else:
                # If it was possible to get the length and it to the
                # appropriate table.
                tables[obj_len][str(path).replace("/", ".")] = obj

    # Convert the default dict to a normal dict and return it.
    return dict(tables)


def walker(obj, path: PurePosixPath) -> Iterable[Tuple[Any, PurePosixPath]]:
    """
    Generator that walks through an object hierarchy and yields the end points of the
    hierarchy.

    :param obj: Root element of the object hierarchy.
    :param path: Path of the root element.
    """
    # If the object is a dictionary, list, etc, yield from the appropriate walker.
    # If the object id a numpy array yield from the ndarray processor to split
    # ND arrays into multiple 1D arrays.
    # If it's anything else, yield it.

    # If the object is a dictionary yield from the dictionary walker.
    if isinstance(obj, (dict, OrderedDict)):
        yield from dict_walker(obj, path)
        return

    # If the object is a list, check if it only contains scalar elements.
    # If is only contains scalar elements yield it.
    # Otherwise yield from the list walker.
    if isinstance(obj, (list, tuple)):
        if all(map(np.isscalar, obj)):
            yield obj, path
        else:
            yield from list_walker(obj, path)
        return

    # If the object is a numpy array yield from the ndarray processor.
    if isinstance(obj, np.ndarray):
        yield from process_ndarray(obj, path)
    else:
        # If it's anything else, yield it.
        yield obj, path


def dict_walker(d: Union[dict, OrderedDict], path: PurePosixPath) -> Iterable[Tuple[Any, PurePosixPath]]:
    """
    Generator that walks through an object hierarchy whose root element is a dictionary
    and yields the end points of the hierarchy.

    :param d: Root dictionary of the object hierarchy.
    :param path: Path of the root dictionary.
    :return:
    """
    # Loop through all items in the dictionary and yield from the "generic" walker.
    for key, obj in d.items():
        yield from walker(obj, path / str(key))


def list_walker(l: Union[list, tuple], path: PurePosixPath) -> Iterable[Tuple[Any, PurePosixPath]]:
    """
        Generator that walks through an object hierarchy whose root element is a list
        or tuple and yields the end points of the hierarchy.

        :param l: Root list/tuple of the object hierarchy.
        :param path: Path of the root list/tuple.
        :return:
        """
    # Loop through all items in the list and yield from the "generic" walker.
    for i, obj in enumerate(l):
        yield from walker(obj, path / f"[{i}]")


def process_ndarray(array: np.ndarray, path: PurePosixPath) -> Iterable[Tuple[Any, PurePosixPath]]:
    """
    Yields 1D :class:`numpy.ndarray` and path object for each higher dimension index
    combination w.r.t the first dimension.

    e.g.

    .. code-block:: python

       a = np.array([[1, 2, 3]
                     [4, 5, 6]
                     [7, 8, 9]])

       p = PurePosixPath("name")

       for obj, path in process_ndarray(a, p):
           print(path, obj)

    Output

    .. code-block:: none

       name_0 [1 4 7]
       name_1 [2 5 8]
       name_2 [3 6 9]


    :param array: Numpy array of N dimensions.
    :param path: Path object with the path to the numpy array in the object hierarchy.
    """
    # If it only has one dimension, return the array and path without any processing.
    if array.ndim == 1:
        yield array, path
    else:
        # Loop through all combinations of higher dimension indices.
        for idx in product(*map(range, array.shape[1:])):
            # Yield all elements for this higher dimension index w.r.t. to the first dimension.
            # e.g. array[:, 1, 2, 3, 4]
            yield array[(slice(None), *idx)], path.with_name(f"{path.name}_{'_'.join(map(str, idx))}")
