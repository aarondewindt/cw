import numpy as np
from scipy.linalg import expm
from typing import Tuple


def c2d(a: np.ndarray, b: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discretizes a continues state space model.

    :param a: State space A matrix
    :param b: State space b matrix
    :param dt: Time step
    :return: Tuple with the discreet A and B matrices.
    """
    m, na = a.shape
    m, nb = b.shape

    s = expm(
        np.vstack((
            np.hstack((a, b)) * dt,
            np.zeros((nb, na+nb))
        ))
    )

    return s[:na, :na], s[:na, na:na+nb]

