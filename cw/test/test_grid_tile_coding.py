import unittest

import numpy as np
import numpy.testing as npt

from cw.tile_coding import GridTileCoding, TileSelection, Tile


class TestGridTileCoding(unittest.TestCase):
    def test_init(self):
        gtc = GridTileCoding(
            center_coordinate=[-2, 1],
            tile_size=[1, 3],
            n_tilings=3,
            initial_value=[0, 1],
            random_offsets=False,
        )

        npt.assert_allclose(gtc.center_coordinates, [[-2., 1.],
                                                     [-1.66666667, 2.],
                                                     [-1.33333333, 3.]])

    def test_get_value(self):
        gtc = GridTileCoding(
            center_coordinate=[0, 0],
            tile_size=[1, 3],
            n_tilings=3,
            initial_value=0,
            random_offsets=False,
        )

        tile_collection = gtc[0.4, 1]
        self.assertIsInstance(tile_collection, TileSelection)

    def test_tile_collection(self):
        tiles = [Tile(np.array([0, 1])), Tile(np.array([1, 2])), Tile(np.array([3, 4]))]
        tc = TileSelection(tiles, [])
        npt.assert_allclose(tc.value, [4/3, 7/3])

    def test_tile_collection_setting(self):
        tiles = [Tile(np.array([0, 1])), Tile(np.array([1, 2])), Tile(np.array([3, 4]))]
        tc = TileSelection(tiles, [])
        tc.value = [9, 10]

        npt.assert_allclose(tc.value, [9, 10])
        npt.assert_allclose(tiles[0].weight, [3., 3.33333333])
        npt.assert_allclose(tiles[1].weight, [3., 3.33333333])
        npt.assert_allclose(tiles[2].weight, [3., 3.33333333])


if __name__ == '__main__':
    unittest.main()
