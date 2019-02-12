import re
from pathlib import PurePosixPath
from itertools import groupby, islice
import numpy as np

from typing import Dict, Union, Any
from cw.tree_node import TreeNode


# Regular expressions matching the naming scheme of ndarrays.
ndarray_re = re.compile(r"^\s*(\w+?)((?:_\d+)+)\s*$")


def tables_to_object_hierarchy(tables):
    """
    Transforms a dictionary of tables into a object hierarchy.

    List elements with the same name ending with trailing integers separated
    by underscores (example `name_0_1`) are combined into a single
    :class:`numpy.ndarray` with n+1 dimensions, where n is the number of trailing
    integers. The first dimension's size is the same as the length of the lists.
    The trailing integers then define the index of the higher dimensions where
    the data will be inserted to.

    Elements have be grouped in sub-dictionaries. If the key of the dictionary
    is a string it will be appended to the front of the child elements name.
    It can thus be used to define namespaces. Otherwise the key is ignored.

    Example
    .. code-block:: python

       inp = {
           4: {
               'foo.quux_0_0': np.array([9, 11, 14, 17]),
               'foo.quux_1_0': np.array([10, 12, 15, 18]),
               'foo.quux_2_0': np.array([11, 13, 16, 19])
           },
           "bar": {
               "bas": 123
           }
       }

       {
           'foo': {
               'quux': np.array([[[9.],  [10.], [11.]],
                                 [[11.], [12.], [13.]],
                                 [[14.], [15.], [16.]],
                                 [[17.], [18.], [19.]]])
           },
           "bar": {
               "bas": 123
           }
       }

    :param tables: Dictionary containing dictionaries whose key is the path
       to the element in the resulting object hierarchy.
    :return: Object hierarchy
    """
    data_tables = flatten_tables(tables)
    process_ndarrays(data_tables)
    root_node = TreeNode.from_path_value_pairs(data_tables)
    root_obj = root_node.object_hierarchy()
    return root_obj


def flatten_tables(tables: Dict[Union[int, str], Dict[str, Any]]):
    """
    Returns a list containing tuples with two elements, the first being a
    :class:`pathlib.PurePosixPath` with the path to the value in the final
    object hierarchy and second one being the value. Namespaces are resolved

    :param tables:
    :return:
    """

    flat_tables = {}
    for namespace, local_tables in tables.items():
        # If the namespace is not a string, the element is placed on the root namespace.
        # Everything behind the hashtag is a comment.
        namespace = namespace.split("#")[0].strip() if isinstance(namespace, str) else ""

        for node_name, node_value in local_tables.items():
            path = PurePosixPath(namespace, *node_name.split("."))
            flat_tables[path] = node_value

    return flat_tables


def process_ndarrays(tables):
    # It is not possible to make the changes in the table inside of the
    # main loop because it's not possible to change the length of a
    # dictionary while iterating through it.
    table_changes = []

    for path, group in find_ndarrays(tables):
        # This list will contain the changes that are needed to be made
        # in the table.
        change = [path, [], None]

        # Find shape of ndarray
        # Initializes the size as 0 for all dimensions.
        # Iterates through all of the elements in the array to look for
        # the largest index and sets the size to the index plus 1.
        shape = [0] * len(group[0][1])
        for _, idx in group:
            for i, size in enumerate(idx):
                size += 1
                if size > shape[i]:
                    shape[i] = size

        # The first dimension of the final array should have the same
        # size as the length of the columns.
        shape = (len(tables[group[0][0]]), *shape)

        # Initialize new ndarray
        array = np.empty(shape)

        # Copy the data of the old column and put it in the new array.
        for col_path, idx in group:
            array[(slice(None), *idx)] = np.array(tables[col_path])
            change[1].append(col_path)

        change[2] = array
        table_changes.append(change)

    # Apply the changes to the table.
    for path, old_paths, value in table_changes:
        for old_path in old_paths:
            del tables[old_path]
        tables[path] = value


def find_ndarrays(tables):
    def ndarray_cols():
        for path, value in tables.items():
            match = ndarray_re.match(path.name)
            if match:
                yield path.with_name(match.group(1)), path, \
                      tuple(map(int, islice(match.group(2).split("_"), 1, None)))

    for k, g in groupby(ndarray_cols(), lambda x: x[0]):
        group = []
        for e in g:
            group.append((e[1], e[2]))
        yield k, group
