"""Package that contains a number of functions to create transformation 
matrices between different reference frames.

For more information go to:
http://daresimserver.duckdns.org/docs/general/reference_frames/index.html
"""

import numpy as np
import math




def tr_ci(t, omega_t=7.2921235169904e-5):
    """
    Transformation Matrix from :ref:`sec:F-I` to :ref:`sec:F-C`.

    :param float t: Time (s) to determine the rotation of the Earth.
    :param float omega_t: Earth's rotational rate (rad/s).
    :return: Transformation matrix from Inertial to ECEF frame.
    :rtype: numpy.ndarray
    """
    
    return np.array([[math.cos(omega_t*t),  math.sin(omega_t*t), 0],
                     [-math.sin(omega_t*t), math.cos(omega_t*t), 0],
                     [0,                    0,                   1]])


def tr_ic(t, omega_t=7.2921235169904e-5):
    """
    Transformation Matrix from :ref:`sec:F-C` to  :ref:`sec:F-I`.
    
    :param float t: Time (s) to determine the rotation of the Earth.
    :param float omega_t: Earth's rotational rate (rad/s).
    :return: Transformation matrix from Inertial to ECEF frame.
    :rtype: numpy.ndarray
    """
    
    return tr_ci(t, omega_t).transpose()


def tr_ab(alpha, beta):
    """
    Transformation Matrix from :ref:`sec:F-b` to :ref:`sec:F-a`.

    :param float alpha: Aerodynamic angle of attack [rad].
    :param float beta: Aerodynamic angle of side-slip [rad].
    :return: Transformation matrix from body to aerodynamic frame.
    :rtype: numpy.ndarray
    """
    
    sang = np.sin([alpha, beta])
    cang = np.cos([alpha, beta])
    
    return np.array([[cang[1]*cang[0], sang[1], cang[1]*sang[0]],
                     [-sang[1]*cang[0], cang[1], -sang[1]*sang[0]],
                     [-sang[0], 0, cang[0]]])


def tr_ba(alpha, beta):
    """
    Transformation Matrix from :ref:`sec:F-a` to :ref:`sec:F-b`.

    :param float alpha: Aerodynamic angle of attack [rad].
    :param float beta: Aerodynamic angle of side-slip [rad].
    :return: Transformation matrix from body to aerodynamic frame.
    :rtype: numpy.ndarray
    """
    
    return tr_ab(alpha,beta).transpose()


def tr_ec(tau, delta):
    """Transformation from :ref:`sec:F-C` to the :ref:`sec:F-E`.

    :param float tau: Longitude (radians) from the Greenwich meridian (tau is positive if
       the vehicle position is east of the Greenwich meridian).
    :param float delta: Latitude (radians) from the equator (delta is positive if the vehicle
       location is on the northern hemisphere).
    :returns: Transformation matrix from ECEF to Vehicle carried normal frame.
    :rtype: numpy.ndarray
    """
    
    sang = np.sin([tau, delta])
    cang = np.cos([tau, delta])
    
    return np.array([[-sang[1]*cang[0], -sang[1]*sang[0], cang[1]],
                     [-sang[0], cang[0], 0],
                     [-cang[1]*cang[0], -cang[1]*sang[0], -sang[1]]])


def tr_ce(tau, delta):
    """
    Transformation from :ref:`sec:F-E` to the :ref:`sec:F-C`.

    :param float tau: Longitude (radians) from the Greenwich meridian (tau is positive if
       the vehicle position is east of the Greenwich meridian).
    :param float delta: Latitude (radians) from the equator (delta is positive if the vehicle
       location is on the northern hemisphere).
    :returns: Transformation matrix from ECEF to Vehicle carried normal frame.
    :rtype: numpy.ndarray
    """
    
    return tr_ec(tau, delta).transpose()


def tr_be(psi, theta, phi):
    """Transformation from :ref:`sec:F-E` to the :ref:`sec:F-b`.

    :param float psi: Yaw angle about the Z_E-axis (radians).
    :param float theta: Pitch angle about the Y_E-axis (radians).
    :param float phi: Roll angle about the X_E-axis (radians).
    :returns: Transformation matrix from ECEF to body frame.
    :rtype: numpy.ndarray
    """
    
    sang = np.sin([psi, theta, phi])
    cang = np.cos([psi, theta, phi])
    
    return np.array([[cang[1]*cang[0], cang[1]*sang[0], -sang[1]],
                     [sang[2]*sang[1]*cang[0]-cang[2]*sang[0], sang[2]*sang[1]*sang[0]+cang[2]*cang[0], sang[2]*cang[1]],
                     [cang[2]*sang[1]*cang[0]+sang[2]*sang[0], cang[2]*sang[1]*sang[0]-sang[2]*cang[0], cang[2]*cang[1]]])


def tr_eb(psi, theta, phi):
    """
    Transformation from the :ref:`sec:F-b` to :ref:`sec:F-E`.
    
    :param float psi: Yaw angle about the Z_E-axis (radians).
    :param float theta: Pitch angle about the Y_E-axis (radians).
    :param float phi: Roll angle about the X_E-axis (radians).
    :returns: Transformation matrix from ECEF to body frame.
    :rtype: numpy.ndarray
    """
    
    return tr_be(psi, theta, phi).transpose()


def tr_ae(chi, gamma, mu):
    """Transformation from the :ref:`sec:F-E` to the :ref:`sec:F-a`.

    :param float chi: Aerodynamic heading angle about the Z_E-axis [rad].
    :param float gamma: Aerodynamic flight-path angle about the Y_E-axis [rad].
    :param float mu: Aerodynamic bank angle about the X_a-axis [rad].
    :returns: Transformation matrix from Vehicle carried normal to aerodynamic frame.
    :rtype: numpy.ndarray
    """
    
    sang = np.sin([chi, gamma, mu])
    cang = np.cos([chi, gamma, mu])
    
    return np.array([[cang[1]*cang[0], cang[1]*sang[0], -sang[1]],
                     [-sang[2]*sang[1]*cang[0]-cang[2]*sang[0], -sang[2]*sang[1]*sang[0]+cang[2]*cang[0], -sang[2]*cang[1]],
                     [cang[2]*sang[1]*cang[0]-sang[2]*sang[0], cang[2]*sang[1]*sang[0]+sang[2]*cang[0], cang[2]*cang[1]]])


def tr_ea(chi, gamma, mu):
    """
    Transformation from the :ref:`sec:F-a` to the :ref:`sec:F-E`.
    
    :param float chi: Aerodynamic heading angle about the Z_E-axis [rad].
    :param float gamma: Aerodynamic flight-path angle about the Y_E-axis [rad].
    :param float mu: Aerodynamic bank angle about the X_a-axis [rad].
    :returns: Transformation matrix from Vehicle carried normal to aerodynamic frame.
    :rtype: numpy.ndarray
    """
    
    return tr_ae(chi, gamma, mu).transpose()


def tr_sb(alpha_0):
    """
    Transformation from the body to stability frame.

    :param alpha_0:
    :return:
    """
    ca = math.cos(alpha_0)
    sa = math.sin(alpha_0)
    return np.array([[ca, 0, -sa], [0, 1, 0], [sa, 0, ca]])

def tr_bs(alpha_0):
    """
    Transformation from the stability to body frame.

    :param alpha_0:
    :return:
    """
    return tr_sb(alpha_0).transpose()