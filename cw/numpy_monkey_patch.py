"""
Numpy monkey patch allowing the module to be used as a literal to create new numpy arrays.

note..: This module needs to be imported before numpy or numpy needs to be reimported after
        it in order for it to be effective.

Example: Creates a 2x2 numpy array.
a = np[[1, 2], [3, 4]]
"""

import numpy as np
from collections import Sequence
from types import ModuleType
import sys

#      w  c(..)o   (
#       \__(-)    __)
#           /\   (
#          /(_)___)
#          w /|
#           | \
#          m  m


class PatchedNumpy(ModuleType):
    def __init__(self):
        super().__init__("numpy")

    def __getitem__(self, item):
        if isinstance(item, Sequence):
            return np.array(item)
        if item is Ellipsis:
            return np.array(())
        else:
            return np.array((item,))

old_numpy = np

new_numpy = sys.modules['numpy'] = PatchedNumpy()
new_numpy.__dict__.update(old_numpy.__dict__)
