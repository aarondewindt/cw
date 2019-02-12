from cw.conversions import hgeo_to_hpot
from math import sqrt, exp

__author__ = ['Nikita Sirons']

# TODO: Write unittest for atmo_isa


def atmo_isa(h, T0=292.15, p0=108900, rho0=1.2985,
             layers=(-610, 11000, 20000, 32000, 47000, 51000, 71000, 84852, 400000),
             lapse_rate=(-0.0065, 0, 0.001, 0.0028, 0, -0.0028, -0.002)):
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
    g0 = 9.80665
    R = 287.05287
    gamma = 1.4
    i = 0
    if h > 86500:
        raise ValueError("ISA Atmosphere module works until 86500km", h)
    else:
        temp_cur = T0
        p_cur = p0
        rho_cur = rho0
        # Initial values of atmospheric parameters
        cur_alt = h
        border = True
        while border:
            border = False
            # Detecting corresponding layer to given altitude
            if h > layers[i + 1]:
                border = True
            T_base = temp_cur
            # Change of altitude in current layer.
            delta_alt = (border * (layers[i + 1] - layers[i])) + \
                        (not border) * (cur_alt - layers[i])
            temp_cur += lapse_rate[i] * delta_alt
            # Atmospheric parameters if Temperature is changing.
            if lapse_rate[i] != 0:
                p_cur *= (temp_cur / T_base) ** (-g0 / (lapse_rate[i] * R))
                rho_cur *= (temp_cur / T_base) ** (-g0 / (lapse_rate[i] * R) - 1)
            # Atmospheric parameters if Temperature is constant.
            else:
                p_cur *= exp(-g0 * delta_alt / (R * temp_cur))
                rho_cur *= exp(-g0 * delta_alt / (R * temp_cur))
            # Update layer index
            i += 1
        a_cur = sqrt(R * gamma * temp_cur)
    return temp_cur, p_cur, rho_cur, a_cur
