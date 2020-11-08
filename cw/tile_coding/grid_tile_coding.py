from typing import Sequence
from collections import defaultdict
from functools import partial

import numpy as np

from .tile import Tile, TileSelection


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

