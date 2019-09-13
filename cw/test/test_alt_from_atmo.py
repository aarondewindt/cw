__author__ = 'Aaron M. de Windt'


import unittest
from cw.atmo import alt_from_rho
from cw.atmo import atmo_isa


class TestAltFromAtmo(unittest.TestCase):
    def test_alt_from_rho(self):
        for h in range(0, 80000, 5000):
            with self.subTest(h=h):
                self.assertAlmostEqual(alt_from_rho(atmo_isa(h)[2]), h)


if __name__ == '__main__':
    unittest.main()
