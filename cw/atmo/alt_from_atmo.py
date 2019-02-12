__author__ = 'Aaron M. de Windt'

from scipy.optimize import newton
from cw.atmo.atmo_isa import atmo_isa

from typing import Callable, Tuple, Optional


def alt_from_temp(temp: float,
                  atmo_model: Optional[Callable[[float], Tuple[float, float, float, float]]]=None) -> float:
    """
    Calculates the altitude from a given atmospheric model.

    ..warning: This function will return only one of the possible solutions.

    :param temp: Atmospheric temperature.
    :param atmo_model: Atmospheric model to use. This must be a callable that takes the
                       altitude as its only parameter and returns a tuple whose first
                       element is the atmospheric temperature.
                       By default this is the ISA standard atmosphere.
    :returns: The altitude for the given temperature
    """
    # TODO: Look for all possible solutions in alt_from_temp.
    atmo_model = atmo_model or atmo_isa
    return newton(lambda h: atmo_model(h)[0] - temp, 0)


def alt_from_rho(rho: float, atmo_model: Optional[Callable[[float], Tuple[float, float, float, float]]]=None) -> float:
    """
    Calculates the altitude from a given atmospheric model.

    :param rho: Atmospheric density.
    :param atmo_model: Atmospheric model to use. This must be a callable that takes the
                       altitude as its only parameter and returns a tuple whose third
                       element is the atmospheric density.
                       By default this is the ISA standard atmosphere.
    :returns: The altitude for the given density
    """
    atmo_model = atmo_model or atmo_isa
    return newton(lambda h: atmo_model(h)[2] - rho, 0)


def alt_from_p(p: float, atmo_model: Optional[Callable[[float], Tuple[float, float, float, float]]]=None) -> float:
    """
    Calculates the altitude from a given atmospheric model.

    :param p: Atmospheric pressure.
    :param atmo_model: Atmospheric model to use. This must be a callable that takes the
                       altitude as its only parameter and returns a tuple whose second
                       element is the atmospheric density.
                       By default this is the ISA standard atmosphere.
    :returns: The altitude for the given density
    """
    atmo_model = atmo_model or atmo_isa
    return newton(lambda h: atmo_model(h)[1] - p, 0)


def alt_from_a(a: float, atmo_model: Optional[Callable[[float], Tuple[float, float, float, float]]]=None) -> float:
    """
    Calculates the altitude from a given atmospheric model.

    ..warning: This function will return only one of the possible solutions.

    :param a: Speed of sound.
    :param atmo_model: Atmospheric model to use. This must be a callable that takes the
                       altitude as its only parameter and returns a tuple whose fourth
                       element is the speed of sound.
                       By default this is the ISA standard atmosphere.
    :returns: The altitude for the given density
    """
    atmo_model = atmo_model or atmo_isa
    return newton(lambda h: atmo_model(h)[3] - a, 0)
