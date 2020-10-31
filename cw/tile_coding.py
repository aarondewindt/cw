from typing import Sequence
from collections import defaultdict
from functools import partial

import numpy as np

from cw.cached import cached


class GridTileCoding:
    def __init__(self,
                 center_coordinate: Sequence[float],
                 tile_size: Sequence[float],
                 n_tilings: int,
                 initial_value,
                 random_offsets: bool=False
                 ):
        center_coordinate = np.asarray(center_coordinate).flatten()
        n_dims = len(tile_size)
        self.center_coordinates = np.empty((n_tilings, n_dims))
        self.tile_size = np.asarray(tile_size).flatten()
        self.n_tilings = n_tilings

        if not np.isscalar(initial_value):
            initial_value = np.asarray(initial_value)

        for idx_tiling in range(n_tilings):
            if random_offsets:
                tiling_offset = np.random.rand(n_dims)
            else:
                tiling_offset = np.ones((n_dims,)) * idx_tiling / n_tilings

            tiling_offset = self.tile_size * tiling_offset
            self.center_coordinates[idx_tiling, :] = center_coordinate + tiling_offset

        self.tilings = [defaultdict(partial(Tile, initial_value / n_tilings)) for _ in range(n_tilings)]

    def __getitem__(self, coordinate) -> 'TileSelection':
        idxs = np.floor((coordinate - self.center_coordinates) / self.tile_size)
        tiles = [self.tilings[i][tuple(idxs[i, :])] for i in range(self.n_tilings)]
        return TileSelection(tiles, coordinate)


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
        return np.average(np.vstack([tile.weight for tile in self.tiles]), axis=0)

    @property
    def is_scalar(self):
        return self.tiles[0].is_scalar

    @value.setter
    def value(self, value):
        new_weight = (value if self.is_scalar else np.asarray(value)) / len(self.tiles)
        for tile in self.tiles:
            tile.weight = new_weight

