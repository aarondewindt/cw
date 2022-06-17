from math import acos, atan2, cos, sin, sqrt, pi, tan, atan, fmod
from numba import jit
from typing import Sequence, Optional

import numpy as np
from numpy.linalg import norm
from scipy.optimize import root_scalar

from .constants import mu_earth, eps64

# Cartesian frame unit vectors
x_hat = np.array([1., 0., 0.])
y_hat = np.array([0., 1., 0.])
z_hat = np.array([0., 0., 1.])

zero_hat = np.array([0., 0., 0.])


@jit(nopython=True)
def limit_zero_2pi(angle):
    """
    Limit an angle such that it's between 0 and 2pi
    """
    return np.fmod(angle + 2 * np.pi, 2 * np.pi)


def cartesian_to_kepler(r: Sequence[float], v: Sequence[float], mu: float = mu_earth):
    """
    Convert Cartesian components to Kepler elements.

    Source:
        Wertz, James Richard. "Mission geometry; orbit and constellation
        design and management." Space Technology Library (2001).

    :param r: Position vector.
    :param v: Velocity vector.
    :param mu: Optional, Standard gravitational parameter, by defaults it's Earth's (3.98600441e14).
    :return: Tuple with the semi-major axis (a), eccentricity (e), inclination (i),
             right ascension of the ascending node (Ω), argument of periapsis (ω), true anomaly (θ).
             eccentric anomaly (E) and the mean anomaly (M).
    """
    # Make sure the arguments are numpy arrays.
    r = np.asarray(r)
    v = np.asarray(v)

    r_norm = norm(r)

    # Orbit angular momentum
    h = np.cross(r, v)
    h_hat = h / norm(h)

    # Ascending node
    n = np.cross(z_hat, h_hat)
    n_norm = norm(n)
    # In case of zero inclination there is no acending node and this
    # cross product will return a zero vector.
    # This is because this particular orientation results in a gibal lock
    # between the RAAN and argument of periapsis.
    # For this case we will define both these angles as zero.
    if n_norm > eps64:
        n_hat = n / n_norm
    else:
        n_hat = None

    # Eccentricity vector and eccentricity
    e_vec = np.cross(v, h) / mu - r / r_norm
    e = norm(e_vec)

    # Semi major axis
    a = 1 / (2 / r_norm - norm(v)**2/mu)

    # Inclination
    i = acos(h[2] / norm(h))

    # Right Ascension of the Ascending Node
    if n_hat is not None:
        raan = atan2(n[1], n[0])
        raan = limit_zero_2pi(raan)
    else:
        raan = 0

    # Argument of periapsis
    # There is no periapsis if the orbit is circular, so set it to zero.
    if n_hat is not None:
        e_hat = (e_vec / norm(e_vec))
        omega = acos(e_hat @ n_hat)
        if (np.cross(n_hat, e_vec) @ h) < 0:
            # omega is between pi and 2pi
            omega = 2 * pi - omega
        omega = limit_zero_2pi(omega)
    else:
        e_hat = (e_vec / norm(e_vec))
        ny_hat = [0, -1, 0]
        omega = acos(e_hat @ ny_hat)
        if (np.cross(ny_hat, e_vec) @ h) < 0:
            # omega is between pi and 2pi
            omega = 2 * pi - omega
        omega = limit_zero_2pi(omega)

    # Anomalies
    true_anomaly = acos(r @ e_vec / (e * r_norm))
    if np.cross(e_vec, r) @ h < 0:
        # true_anomaly is between pi and 2pi
        true_anomaly = 2 * pi - true_anomaly
    true_anomaly = limit_zero_2pi(true_anomaly)

    eccentric_anomaly = atan2(sqrt(1 - e**2) * sin(true_anomaly), e + cos(true_anomaly))
    eccentric_anomaly = limit_zero_2pi(eccentric_anomaly)

    mean_anomaly = eccentric_anomaly - e * sin(eccentric_anomaly)
    mean_anomaly = limit_zero_2pi(mean_anomaly)

    return a, e, i, raan, omega, true_anomaly, eccentric_anomaly, mean_anomaly


@jit(nopython=True)
def cartesian_to_kepler_no_anomalies(r: Sequence[float], v: Sequence[float], mu: float = mu_earth):
    """
    Convert Cartesian components to Kepler elements. Accelerated with numba, doesn't return anomalies.

    Source:
        Wertz, James Richard. "Mission geometry; orbit and constellation
        design and management." Space Technology Library (2001).

    :param r: Position vector.
    :param v: Velocity vector.
    :param mu: Optional, Standard gravitational parameter, by defaults it's Earth's (3.98600441e14).
    :return: Tuple with the semi-major axis (a), eccentricity (e), inclination (i),
             right ascension of the ascending node (Ω), argument of periapsis (ω).
    """
    r_norm = norm(r)

    # Orbit angular momentum
    h = np.cross(r, v)
    h_hat = h / norm(h)

    # Ascending node
    n = np.cross(z_hat, h_hat)
    n_norm = norm(n)
    # In case of zero inclination there is no acending node and this
    # cross product will return a zero vector.
    # This is because this particular orientation results in a gibal lock
    # between the RAAN and argument of periapsis.
    # For this case we will define both these angles as zero.

    if n_norm > eps64:
        is_circular = False
        n_hat = n / n_norm
    else:
        is_circular = True
        n_hat = zero_hat

    # Eccentricity vector and eccentricity
    e_vec = np.cross(v, h) / mu - r / r_norm
    e = norm(e_vec)

    # Semi major axis
    a = 1. / (2. / r_norm - norm(v)**2/mu)

    # Inclination
    i = acos(h[2] / norm(h))

    # Right Ascension of the Ascending Node
    if not is_circular:
        raan = atan2(n[1], n[0])
        raan = limit_zero_2pi(raan)
    else:
        raan = 0.

    # Argument of periapsis
    # There is no periapsis if the orbit is circular, so set it to zero.
    if is_circular:
        omega = 0.
    else:
        e_hat = (e_vec / norm(e_vec))
        omega = acos(e_hat @ n_hat)
        if (np.cross(n_hat, e_vec) @ h) < 0.:
            # omega is between pi and 2pi
            omega = 2. * pi - omega
        omega = limit_zero_2pi(omega)

    return a, e, i, raan, omega


@jit(nopython=True)
def cartesian_to_kepler_no_anomalies_2d(r2d: Sequence[float], v2d: Sequence[float], mu: float = mu_earth):
    """
    Convert Cartesian components to Kepler elements. Accelerated with numba, doesn't return anomalies.

    Source:
        Wertz, James Richard. "Mission geometry; orbit and constellation
        design and management." Space Technology Library (2001).

    :param r: Position vector.
    :param v: Velocity vector.
    :param mu: Optional, Standard gravitational parameter, by defaults it's Earth's (3.98600441e14).
    :return: Tuple with the semi-major axis (a), eccentricity (e), inclination (i),
             right ascension of the ascending node (Ω), argument of periapsis (ω).
    """
    r = np.append(r2d, 0.)
    v = np.append(v2d, 0.)

    r_norm = norm(r)

    # Orbit angular momentum
    h = np.cross(r, v)

    # Eccentricity vector and eccentricity
    e_vec = np.cross(v, h) / mu - r / r_norm
    e = norm(e_vec)

    # Semi major axis
    a = 1. / (2. / r_norm - norm(v)**2/mu)

    return a, e


def kepler_to_cartesian(a: float, e: float, i: float, raan: float, omega: float,
                        true_anomaly: Optional[float] = None,
                        eccentric_anomaly: Optional[float] = None,
                        mean_anomaly: Optional[float] = None,
                        mu: float = mu_earth):
    """
    Convert to Kepler elements to Cartesian components. At least one of the anomalies is required,
    the rest can be set to `None`.

    Source:
        Wertz, James Richard. "Mission geometry; orbit and constellation
        design and management." Space Technology Library (2001).

    :param a: Semi-major axis.
    :param e: Eccentricity.
    :param i: Inclination.
    :param raan: Right ascension of the ascending node (Omega).
    :param omega: Argument of periapsis.
    :param true_anomaly: True anomaly.
    :param eccentric_anomaly: Eccentric anomaly.
    :param mean_anomaly: Mean anomaly.
    :param mu: Optional, Standard gravitational parameter, by defaults it's Earth's (3.98600441e14).
    :return: Tuple with the cartesian position (r) and velocity (V).
    """

    # We need the true anomaly for the calculations, check if it was given.
    if true_anomaly is None:
        # If not we can calculate it from the eccentric anomaly, check if it was given.
        if eccentric_anomaly is None:
            # If not we can calculate the eccentric anomaly from the mean anomaly.
            # Check if it was given.
            if mean_anomaly is None:
                # We need at least one of the three anomalies. Raise error.
                raise ValueError('At least one of "True anomaly", "Eccentric anomaly" or "Mean anomaly" '
                                 'is required')
            else:
                eccentric_anomaly = eccentric_anomaly_from_mean_anomaly(e, mean_anomaly)

        # We should have a value for the eccentric anomaly at this point.
        true_anomaly = true_anomaly_from_eccentric_anomaly(e, eccentric_anomaly)

    # Semiparameter
    p = a * (1 - e**2)

    # Position in the perifocal coordinate system (pf)
    r_pf = np.zeros(3)
    r_pf[0] = p * cos(true_anomaly) / (1 + e * cos(true_anomaly))
    r_pf[1] = p * sin(true_anomaly) / (1 + e * cos(true_anomaly))

    # Velocity in the perifocal coordinate system
    v_pf = np.zeros(3)
    v_pf[0] = -sqrt(mu / p) * sin(true_anomaly)
    v_pf[1] = sqrt(mu / p) * (e + cos(true_anomaly))

    # Transformation matrix from the perifocal to inertial coordinate system.
    cr = cos(raan)
    co = cos(omega)
    ci = cos(i)
    sr = sin(raan)
    so = sin(omega)
    si = sin(i)

    c_pf = np.array([[cr*co - sr*so*ci, -cr*so - sr*co*ci, sr*si],
                     [sr*co + cr*so*ci, -sr*so + cr*co*ci, -cr*si],
                     [so*si, co*si, ci]])

    # Position and velocity in the inertial frame
    position = c_pf @ r_pf
    velocity = c_pf @ v_pf

    return position, velocity


def eccentric_anomaly_from_mean_anomaly(e: float, mean_anomaly: float):
    """
    Calculates the eccentric anomaly from the mean anomaly using a
    numerical root-finding method.

    :param e: Eccentricity.
    :param mean_anomaly: Mean anomaly.
    :return: Mean anomaly
    """
    # Equation to solve. This is the equation to calculate the mean
    # anomaly from the eccentric anomaly minus the mean anomaly.
    # So the solution will be at the root.
    def f(e_ano):
        return e_ano - e * sin(e_ano) - mean_anomaly

    # Solve it numerically, we're letting scipy choose a method for us here.
    solution = root_scalar(f, x0=mean_anomaly, bracket=[0, 2 * pi])
    return solution.root


def true_anomaly_from_eccentric_anomaly(e: float, eccentric_anomaly):
    """
    Calculates the true anomaly from the eccentric anomaly

    :param e: Eccentricity.
    :param eccentric_anomaly: Eccentric anomaly.
    :return: True anomaly
    """
    numerator = sqrt(1+e) * sin(eccentric_anomaly/2)
    denominator = sqrt(1-e) * cos(eccentric_anomaly/2)
    return 2 * atan2(numerator, denominator)

#     return 2 * atan(sqrt((1+e)/(1-e)) * tan(eccentric_anomaly / 2))
