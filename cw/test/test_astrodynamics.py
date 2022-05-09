import unittest
import numpy as np
import numpy.testing as npt
from tabulate import tabulate

from ..astrodynamics import cartesian_to_kepler, kepler_to_cartesian, cartesian_to_kepler_no_anomalies_2d, \
    eccentric_anomaly_from_mean_anomaly, true_anomaly_from_eccentric_anomaly, limit_zero_2pi


class TestAstrodynamics(unittest.TestCase):
    def setUp(self) -> None:
        self.print_results = False
        self.print_results_cartesian_to_kepler = False
        self.print_results_kepler_to_cartesian__mean_anomaly = False

        # Lecture slide examples.
        self.example_data = [
            {
                "cartesian": ([-2700816.14, -3314092.80, 5266346.42], [5168.606550, -5597.546618, -868.878445]),
                "kepler": (6787746.891, 0.000731104,
                           *np.radians((51.68714486, 127.5486706, 74.21987137, 24.10027677, 24.08317766, 24.06608426)))
            },
            {
                "cartesian": ([3126974.99, -6374445.74, 28673.59], [-254.91197, -83.30107, 7485.70674]),
                "kepler": (7096137.00, 0.0011219,
                           *np.radians((92.0316, 296.1384, 120.6878, 239.5437, 239.5991, 239.6546)))
            }
        ]

    def test_cartesian_to_kepler(self):
        """
        Tests the cartesian_to_kepler(...) function using the examples in the lecture slides.
        """
        results = []
        for example_data in self.example_data:
            kepler = cartesian_to_kepler(*example_data['cartesian'])
            results.append(kepler)
            npt.assert_allclose(kepler, example_data['kepler'], rtol=1e-6)

        if self.print_results and self.print_results_cartesian_to_kepler:
            first_col = np.array([[r"$a\ [m]$", r"$e\ [-]$", r"$i\ [\degree]$", r"$\Omega\ [\degree]$",
                                   r"$\omega\ [\degree]$", r"$\theta\ [\degree]$", r"$E\ [\degree]$",
                                   r"$M\ [\degree]$"]]).T
            headers = ["", "Result", "Expected", "Error\%"]
            for result, expected in zip(results, self.example_data):
                result = np.array(result).reshape((8, 1))
                expected = np.array(expected['kepler']).reshape((8, 1))
                result[2:] *= 180 / np.pi
                expected[2:] *= 180 / np.pi
                error = abs(result - expected) / expected * 100
                table = np.hstack((first_col, result, expected, error))
                print(r"\begin{table}[h!] \centering")
                print(tabulate(table, headers, tablefmt="latex_raw"))
                print(r"\end{table}")

    def test_kepler_to_cartesian__true_anomaly(self):
        """
        Tests the kepler_to_cartesian(...) function using the examples in the lecture slides by
        only giving it the true anomaly.
        """
        for example_data in self.example_data:
            cartesian = kepler_to_cartesian(*(example_data['kepler'][:6]))
            npt.assert_allclose(cartesian, example_data['cartesian'], rtol=1e-3)

    def test_kepler_to_cartesian__eccentric_anomaly(self):
        """
        Tests the kepler_to_cartesian(...) function using the examples in the lecture slides by
        only giving it the eccentric anomaly.
        """
        for example_data in self.example_data:
            cartesian = kepler_to_cartesian(*(example_data['kepler'][:5]), eccentric_anomaly=example_data['kepler'][6])
            npt.assert_allclose(cartesian, example_data['cartesian'], rtol=1e-3)

    def test_kepler_to_cartesian__mean_anomaly(self):
        """
        Tests the kepler_to_cartesian(...) function using the examples in the lecture slides by
        only giving it the mean anomaly.
        """
        results = []
        for example_data in self.example_data:
            cartesian = kepler_to_cartesian(*(example_data['kepler'][:5]), mean_anomaly=example_data['kepler'][7])
            results.append(cartesian)
            npt.assert_allclose(cartesian, example_data['cartesian'], rtol=1e-3)

        if self.print_results and self.print_results_kepler_to_cartesian__mean_anomaly:
            first_col = np.array([[
                r"$x\ [m]$", r"$y\ [m]$", r"$z\ [m]$", r"$\dot{x}\ [m\ s^{-1}]$",
                r"$\dot{y}\ [m\ s^{-1}]$", r"$\dot{z}\ [m\ s^{-1}]$"]]).T

            headers = ["", "Result", "Expected", "Error\%"]
            for result, expected in zip(results, self.example_data):
                result = np.array(result).reshape((6, 1))
                expected = np.array(expected['cartesian']).reshape((6, 1))
                error = abs(result - expected) / expected * 100
                table = np.hstack((first_col, result, expected, error))
                print(r"\begin{table}[h!] \centering")
                print(tabulate(table, headers, tablefmt="latex_raw"))
                print(r"\end{table}")

    def test_eccentric_anomaly_from_mean_anomaly(self):
        for example_data in self.example_data:
            e = example_data["kepler"][1]
            mean_anomaly = example_data["kepler"][7]
            eccentric_anomaly = eccentric_anomaly_from_mean_anomaly(e, mean_anomaly)
            expected_eccentric_anomaly = example_data['kepler'][6]
            npt.assert_allclose(eccentric_anomaly, expected_eccentric_anomaly, rtol=1e-3)

    def test_true_anomaly_from_eccentric_anomaly(self):
        for example_data in self.example_data:
            e = example_data["kepler"][1]
            eccentric_anomaly = example_data["kepler"][6]
            true_anomaly = true_anomaly_from_eccentric_anomaly(e, eccentric_anomaly)
            expected_true_anomaly = example_data['kepler'][5]
            npt.assert_allclose(true_anomaly, expected_true_anomaly, rtol=1e-3)

    def test_kepler_to_cartesian_to_kepler(self):
        """
        Tests specific cases by converting from Kepler to Cartesian and back and checking
        the new kepler elements against the original.
        """
        kepler_elements_list = [
            # Semimajor axis, eccentricity, inclination, raan, omega, true anomaly

            # Argument of perigee between 0 and 180
            # True anomaly between 0 and 180
            (7000000, 0.0008, *np.radians((45., 10., 45., 45))),

            # Argument of perigee between 180 and 360
            (7000000, 0.0008, *np.radians((45., 10., 275., 45))),

            # True anomaly between 180 and 360
            (7000000, 0.0008, *np.radians((45., 10., 45., 275))),

            # Argument of periapsis and true anomaly between 180 and 360
            (7000000, 0.0008, *np.radians((45., 10., 275., 275))),

            # Zero inclination
            (7000000, 0.0008, *np.radians((0., 0., 0., 45))),

            # 90deg inclination
            (7000000, 0.0008, *np.radians((90., 10., 45., 45))),

            # Negative inclination
            (7000000, 0.0008, *np.radians((-45., 10., 45., 45))),
            (7000000, 0.0008, *np.radians((-45., 10., 275., 45))),
            (7000000, 0.0008, *np.radians((-45., 10., 45., 275))),
            (7000000, 0.0008, *np.radians((-45., 10., 275., 275))),
            (7000000, 0.0008, *np.radians((-90., 10., 45., 45))),

            # Highly eccentric orbits
            (7000000, 0.999, *np.radians((45., 10., 45., 45))),
            (7000000, 0.999, *np.radians((45., 10., 275., 45))),
            (7000000, 0.999, *np.radians((45., 10., 45., 275))),
            (7000000, 0.999, *np.radians((45., 10., 275., 275))),
            (7000000, 0.999, *np.radians((90., 10., 45., 45))),
            (7000000, 0.999, *np.radians((-45., 10., 45., 45))),
            (7000000, 0.999, *np.radians((-45., 10., 275., 45))),
            (7000000, 0.999, *np.radians((-45., 10., 45., 275))),
            (7000000, 0.999, *np.radians((-45., 10., 275., 275))),
            (7000000, 0.999, *np.radians((-90., 10., 45., 45))),
        ]

        for i, kepler in enumerate(kepler_elements_list):
            with self.subTest(i=i):
                cartesian_results = kepler_to_cartesian(*kepler)
                result_kepler = cartesian_to_kepler(*cartesian_results)

                # Select all elements but the eccentric and mean anomaly,
                # we assume these are calculated correctly in this test.
                result_kepler = list(result_kepler[:6])

                if kepler[2] < 0:
                    # The cartesian_to_kepler(...) function doesn't
                    # know whether a negative inclination was passed to
                    # the kepler_to_cartesian(...) function or not, so
                    # is always assumes a positive inclination.
                    # This means that the RAAN will be at the opposite
                    # side of the orbit.
                    # In order to compare the results for these test cases
                    # the following corrections are done.

                    # Make this inclination negative
                    result_kepler[2] *= -1

                    # Rotate the RAAN by 180deg
                    result_kepler[3] += np.pi
                    result_kepler[3] = limit_zero_2pi(result_kepler[3])

                    # Rotate the argument of periapsis by 180 deg
                    result_kepler[4] += np.pi
                    result_kepler[4] = limit_zero_2pi(result_kepler[4])

                npt.assert_allclose(result_kepler, kepler, rtol=1e-6)
    
    def test_cartesian_to_kepler_no_anomalies_2d(self):
        cartesian = (np.array([-2700816.14, -3314092.80]), np.array([5168.606550, -5597.546618]))
        kepler = (3103819.8260330744, 0.3993462842127281)
        
        result_kepler = cartesian_to_kepler_no_anomalies_2d(*cartesian)
        npt.assert_allclose(result_kepler, kepler, rtol=1e-6)
        