import unittest
from itertools import product
from math import cos, sin

import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt

from cw.tile_coding import IntTileCoding, TileSelection, Tile


class TestGridTileCoding(unittest.TestCase):
    def test_init(self):
        gtc = IntTileCoding(
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
        gtc = IntTileCoding(
            center_coordinate=[0, 0],
            tile_size=[1, 3],
            n_tilings=3,
            initial_value=0,
            random_offsets=False,
        )

        tile_collection = gtc[0.4, 1]
        self.assertIsInstance(tile_collection, TileSelection)

    def test_int_coords(self):
        gtc = IntTileCoding(
            center_coordinate=[0, 0],
            tile_size=[1, 3],
            n_tilings=3,
            initial_value=0,
            random_offsets=False,
        )

        tc = gtc[0., 0, 0]
        tc.value = 2

        tc = gtc[0.4, 0, 0]
        tc = gtc[0., 1, 0]
        tc = gtc[0., 1, 2]

    def test_exp1(self):
        gtc = IntTileCoding(
            center_coordinate=[0, 0],
            tile_size=[1, 1],
            n_tilings=10,
            initial_value=0,
            random_offsets=False,
        )

        x_range = (-10, 10)
        y_range = (-10, 10)

        xs = np.linspace(*x_range, 50)
        ys = np.linspace(*y_range, 50)
        for x, y, is_cos in product(xs, ys, [0, 1]):
            # x = np.random.uniform(*x_range)
            # y = np.random.uniform(*y_range)
            if is_cos:
                gtc[x, y, 0].value = y * cos(x)
            else:
                gtc[x, y, 1].value = sin(y * x)

        size = 500
        xs = np.linspace(*x_range, size)
        ys = np.linspace(*y_range, size)
        m1 = np.empty((size, size))
        m2 = np.empty((size, size))
        for (ix, x), (iy, y) in product(enumerate(xs), enumerate(ys)):
            m1[ix, iy] = gtc[x, y, 0].value
            m2[ix, iy] = gtc[x, y, 1].value

        plt.figure()
        plt.imshow(m1)

        plt.figure()
        plt.imshow(m2)

        plt.show()


if __name__ == '__main__':
    unittest.main()
