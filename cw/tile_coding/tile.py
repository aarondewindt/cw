from typing import Sequence

import numpy as np

from cw.cached import cached


class Tile:
    def __init__(self, initial_weight):
        self.is_scalar = np.isscalar(initial_weight)
        self.weight = initial_weight

    def __repr__(self):
        return f"<Tile {self.weight}>"


class TileSelection:
    def __init__(self, tiles: Sequence[Tile], coordinate):
        self.tiles = tiles
        self.coordinate = coordinate

    @cached
    def value(self):
        return np.sum(np.vstack([tile.weight for tile in self.tiles]), axis=0)

    @property
    def is_scalar(self):
        return self.tiles[0].is_scalar

    @value.setter
    def value(self, value):
        new_weight = (value if self.is_scalar else np.asarray(value)) / len(self.tiles)
        for tile in self.tiles:
            tile.weight = new_weight
