from math import log, sin, cos
import numpy as np

alt_20_ft = 6.096
alt_1000_ft = 304.8
m_to_ft = 3.28084


# Reference: MIL-F-8785C, MILITARY SPECIFICATION: FLYING QUALITIES OF PILOTED AIRPLANES (05 NOV 1980)

def wind_log(alt, speed, heading, z0=2.0, degree=True):
    if degree:
        heading = heading * 0.017453292519943295

    if alt <= 0:
        v = 0
    elif (alt > 0) and (alt < alt_20_ft):
        v = speed / alt_20_ft * alt
    elif (alt >= alt_20_ft) and (alt <= alt_1000_ft):
        v = speed * log(alt * m_to_ft / z0) / log(20 / z0)
    else:
        v = speed * log(1000 / z0) / log(20 / z0)
    _VN = v * cos(heading)
    _VE = v * sin(heading)
    return _VN, _VE


def wind_log_table(speed, heading, z0=2.0):
    alts = (-100, 0, *np.logspace(np.log10(alt_20_ft), np.log10(alt_1000_ft), 20), 10e9)
    vs = [None] * len(alts)

    for i, alt in enumerate(alts):
        if alt <= 0:
            vs[i] = 0
        elif (alt > 0) and (alt < alt_20_ft):
            vs[i] = speed / alt_20_ft * alt
        elif (alt >= alt_20_ft) and (alt <= alt_1000_ft):
            vs[i] = speed * log(alt * m_to_ft / z0) / log(20 / z0)
        else:
            vs[i] = speed * log(1000 / z0) / log(20 / z0)

    return np.array([alts, vs, [heading] * len(alts)]).T


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    t = wind_log_table(5, 180)
    plt.plot(t[:-1, 0], t[:-1, 1])
    plt.show()

