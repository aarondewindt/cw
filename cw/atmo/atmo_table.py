from cw.conversions import hgeo_to_hpot
import scipy.interpolate as interpolate
import math


__author__ = ['Nikita Sirons']

# Write unittest for atmo_table.


def atmo_table(h, atmo_table):
    """
    Parameters
    ----------
    h : float
        Geopotential altitude [m]

    Returns
    -------
    temp_cur : float
        Atmospheric Temperature [K]
    p_cur : float
        Atmospheric Pressure [Pa]
    rho_cur : float
        Atmospheric Density [kg/m^3]
    a_cur : float
        Speed of Sound [m/s]
    """
    # Determine vapure pressure of water with Herman Wobus polynominal
    c0 = 0.99999683
    c1 = -0.90826951 * 10 ** (-2)
    c2 = 0.78736169 * 10 ** (-4)
    c3 = -0.61117958 * 10 ** (-6)
    c4 = 0.43884187 * 10 ** (-8)
    c5 = -0.29883885 * 10 ** (-10)
    c6 = 0.21874425 * 10 ** (-12)
    c7 = -0.17892321 * 10 ** (-14)
    c8 = 0.11112018 * 10 ** (-16)
    c9 = -0.30994571 * 10 ** (-19)

    e_so = 6.1078

    R = 287.05287
    # Air gas constant"
    gamma = 1.4
    # Heat capacity ratio"
    temp_inter = interpolate.interp1d(atmo_table[:, 0], atmo_table[:, 1])
    pressure_inter = interpolate.interp1d(atmo_table[:, 0], atmo_table[:, 2])
    humid_inter = interpolate.interp1d(atmo_table[:, 0], atmo_table[:, 3])

    if h < 1:
        raise ValueError("Altitude is higher")
    else:
        hp = hgeo_to_hpot(h)
        if hp < min(atmo_table[:, 0]):
            hp = min(atmo_table[:, 0])
        elif hp > max(atmo_table[:, 0]):
            hp = max(atmo_table[:, 0])

        relative_humid = humid_inter(hp)
        temp = temp_inter(hp) - 273.15  # temp in degree c
        p_h20 = c0 + temp * (c1 + temp * (c2 + temp * (c3 + temp * (
            c4 + temp * (c5 + temp * (c6 + temp * (c7 + temp * (c8 + temp * c9))))))))
        p = pressure_inter(hp)
        # Total Pressure
        es = e_so / p_h20 **8
        # Saturation pressure of water vapor
        p_v = relative_humid * es
        # Pressure of water vapor
        rho = p / (R * (temp + 273.15)) * (1 - 0.378 * p_v / p)

        temp = temp + 273.15
        a = math.sqrt(R * gamma * (temp + 273.15))

    return temp, p, rho, a