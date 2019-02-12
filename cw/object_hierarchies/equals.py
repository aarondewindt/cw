from collections import OrderedDict
from typing import Union
import numpy as np
from decimal import Decimal
from pathlib import PurePosixPath
from cmath import isclose


def object_hierarchy_equals(obj1, obj2):
    """
    Checks whether two object hierarchies of dictionaries, lists, tuples, etc, match equal other.

    :returns: A list of error messages. The list will be empty if no errors are found.
    """

    # This list will contain strings with the errors found.
    errors = []

    # Loop through the entire object tree. This loop will not
    # return the dictionaries, lists, tuple, etc. The check of
    # those is handled by the walker function.
    for hierarchy_path, value_1, value_2 in walker(obj1, obj2, PurePosixPath("/"), errors):

        # If both values are integers we compare them using the inequality operator.
        if isinstance(value_1, int) and isinstance(value_2, int):
            if value_1 != value_2:
                errors.append(f"Value mismatch at '{hierarchy_path}'")

        # If at least one of the elements is of type Decimal, use the standard isclose
        # function for complex numbers.
        elif isinstance(value_1, Decimal) or isinstance(value_2, Decimal):
            if not isclose(value_1, value_2):
                errors.append(f"Value mismatch at '{hierarchy_path}'")

        # If it is a number, matrix, etc, compare them with the isclose function
        # from numpy.
        elif isinstance(value_1, (float, complex, np.generic, int)):
            if not np.all(np.isclose(value_1, value_2)):
                errors.append(f"Value mismatch at '{hierarchy_path}'")

        # For anything else use the inequality operator.
        else:
            try:
                if value_1 != value_2:
                    errors.append(f"Value mismatch at '{hierarchy_path}'")
            except:
                # It might be that's impossible to use the inequality operator
                # between certain types.
                errors.append(f"Unable to compare values at '{hierarchy_path}'")

    # Return the list of errors.
    return errors


def walker(obj1, obj2, hierarchy_path: PurePosixPath, errors: list):
    """
    Main walker generator. This generator yields all "scalar" elements in the tree.
    So the final elements that don't contain other elements.

    :param obj1: First object to compare.
    :param obj2: Second object to compare
    :param hierarchy_path: Path object with the path to the object.
    :param errors: List onto which to append error messages.
    :return: Generator yielding tuples containing, the path object,
       the object from the first tree and the object from the second tree.
    """
    # If the object is a dictionary yield from the dictionary walker.
    # If the object is a list/tuple, yield from the list walker.
    # If the object is anything else, yield it.
    if isinstance(obj1, (dict, OrderedDict)):
        yield from dict_walker(obj1, obj2, hierarchy_path, errors)
    elif isinstance(obj1, (list, tuple, np.ndarray)):
        yield from list_walker(obj1, obj2, hierarchy_path, errors)
    else:
        yield hierarchy_path, obj1, obj2


def dict_walker(d1: Union[dict, OrderedDict], d2: Union[dict, OrderedDict], hierarchy_path: PurePosixPath, errors: list):
    """
    Walker generator for dictionaries. This generator yields all "scalar" elements in a tree whose
    root element is a dictionary. So the final elements that don't contain other elements.

    :param d1: Dictionary from the first tree.
    :param d2: Dictionary from the second tree.
    :param hierarchy_path: Path object with the path to the dictionary.
    :param errors: List onto which to append error messages.
    :return: Generator yielding tuples containing, the path object,
       the object from the first tree and the object from the second tree.
    """
    # Check whether both input "dictionaries" are actually dictionaries.
    if (not isinstance(d1, (dict, OrderedDict))) or (not isinstance(d2, (dict, OrderedDict))):
        errors.append(f"Type mismatch at '{hierarchy_path}'.")
        return

    # Check if d1 contains all elements in d2.
    for key, value in d2.items():
        if key not in d1:
            errors.append(f"Missing element '{hierarchy_path / str(key)}' in the first object hierarchy.")

    # Loop through each element in d1
    for key, value_1 in d1.items():
        # Create path to the object.
        path = hierarchy_path / str(key)

        # Check whether d2 contains the element.
        if key in d2:
            # If it does, yield from the walker function.
            yield from walker(value_1, d2[key], path, errors)
        else:
            # If not, add an error msg.
            errors.append(f"Missing element '{path}' in the second object hierarchy.")


def list_walker(l1: Union[list, tuple], l2: Union[list, tuple], hierarchy_path: PurePosixPath, errors: list):
    """
    Walker generator for lists, tuples and ndarrays. This generator yields all "scalar" elements in a tree whose root
    element is a list. So the final elements that don't contain other elements.

    :param l1: List from the first tree.
    :param l2: List from the second tree.
    :param hierarchy_path: Path object with the path to the dictionary.
    :param errors: List onto which to append error messages.
    :return: Generator yielding tuples containing, the path object,
       the object from the first tree and the object from the second tree.
    """

    # Check whether the input "lists" are actually lists, tuples or ndarrays.
    if (not isinstance(l1, (list, tuple, np.ndarray))) or (not isinstance(l2, (list, tuple, np.ndarray))):
        errors.append(f"Type mismatch at '{hierarchy_path}.")
        return

    # Check whether the length is the same.
    if len(l1) != len(l2):
        errors.append(f"Sequence length mismatch at '{hierarchy_path}'.")
        return

    # Loop through each element.
    for index, (value_1, value_2) in enumerate(zip(l1, l2)):
        # Create path to the element.
        # If the root object of the tree is a list, then the name of the
        # path object will be empty and will raise a Value error when it's requested.
        # If this is the case. Create a new Path object with the index as it's name.
        try:
            path = hierarchy_path.with_name(hierarchy_path.name + f"[{index:d}]")
        except ValueError:
            path = PurePosixPath(f"/[{index:d}]")

        # Yield from the walker function.
        yield from walker(
            value_1,
            value_2,
            path,
            errors)
