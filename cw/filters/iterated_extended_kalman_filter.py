import numpy as np
import xarray as xr
import sympy as sp
import control as ct

from typing import Callable, Optional, Sequence, Union, Tuple
from collections import deque
from textwrap import indent
from tqdm.auto import tqdm
from scipy.linalg import expm
from IPython.display import display, Markdown
from scipy.signal import cont2discrete


class IteratedExtendedKalmanFilter:
    """
    :param t: Sympy symbol representing time. By default this will be 't'.
    :param x: Sequence of sympy symbols representing the state vector.
    :param u: Sequence of sympy symbols representing the input vector.
    :param z: Sequence of sympy symbols or strings with the name of the measurement vector.
    :param f: Sympy matrix with the state dynamics equations.
    :param fx: Optional sympy matrix with the state dynamics jacobian.
    :param h: Sympy matrix with the observation equations.
    :param hx: Optional sympy matrix with the observation jacobian.
    :param g: Sympy matrix with the system input noise matrix.
    :param max_iterations: Maximum number of iteration per time step.
    :param eps: Maximum iteration error.
    """
    def __init__(self, *,
                 t: sp.Symbol=None,
                 x: Sequence[sp.Symbol],
                 u: Optional[Sequence[sp.Symbol]]=None,
                 z: Sequence[Union[str, sp.Symbol]],
                 f: sp.Matrix,
                 fx: Optional[sp.Matrix]=None,
                 h: sp.Matrix,
                 hx: Optional[sp.Matrix] = None,
                 g: sp.Matrix,
                 max_iterations: Optional[int]=100,
                 eps: Optional[float]=1e-10):
        # Check inputs
        assert max_iterations >= 1

        # Store input
        self.t = t or sp.symbols("t")
        self.x = tuple(x)
        self.u = tuple(u or ())
        self.z = tuple(z)
        self.txu = (self.t, *self.x, *self.u)
        self.f = f
        self.h = h
        self.g = g
        self.max_iterations = max_iterations
        self.eps = eps

        self.t_name = self.t.name
        self.x_names = [x.name for x in self.x]
        self.u_names = [u.name for u in self.u]
        self.z_names = [z.name if isinstance(z, sp.Symbol) else z for z in self.z]

        # Compute jacobians if necessary.
        self.fx = fx if fx else f.jacobian(x)
        self.hx = hx if hx else h.jacobian(x)

        # Create callables from the functions
        self.f_func: Callable = sp.lambdify(self.txu, f)
        self.h_func: Callable = sp.lambdify(self.txu, h)
        self.g_func: Callable = sp.lambdify(self.txu, g)
        self.fx_func: Callable = sp.lambdify(self.txu, self.fx)
        self.hx_func: Callable = sp.lambdify(self.txu, self.hx)

    def __repr__(self):
        return f"IteratedExtendedKalmanFilter:\n" + indent(
            f"x:\n{indent(sp.pretty(self.x, wrap_line=False), '  ')}\n"
            f"z:\n{indent(sp.pretty(self.z, wrap_line=False), '  ')}\n"
            f"u:\n{indent(sp.pretty(self.u, wrap_line=False), '  ')}\n"
            f"f:\n{indent(sp.pretty(self.f, wrap_line=False), '  ')}\n"
            f"g:\n{indent(sp.pretty(self.g, wrap_line=False), '  ')}\n"
            f"h:\n{indent(sp.pretty(self.h, wrap_line=False), '  ')}\n"
            f"fx:\n{indent(sp.pretty(self.fx, wrap_line=False), '  ')}\n"
            f"hx:\n{indent(sp.pretty(self.hx, wrap_line=False), '  ')}\n", '  ')

    def _ipython_display_(self):
        display(Markdown(self.latex(True)))

    def latex(self, jacobians=False):
        def to_latex(name, equation):
            return f"$${name} = {sp.latex(equation)}$$"

        equations = [("x", self.x),
                     ("z", self.z),
                     ("u", self.u),
                     ("f(\dots)", self.f),
                     ("g(\dots)", self.g),
                     ("h(\dots)", self.h)]

        if jacobians:
            equations.extend([('F_x(\dots)', self.fx),
                              ('H_x(\dots)', self.hx)])

        return "  \n".join(to_latex(name, equation) for name, equation in equations)

    def sim(self,
            *,
            x_0: Sequence[float],
            u: xr.Dataset,
            system_noise: Sequence[float],
            system_bias: Sequence[float],
            measurement_noise: Sequence[float],
            measurement_bias: Sequence[float]) -> xr.Dataset:

        # Check inputs.
        assert len(x_0) == len(self.x), "Length of x_0 must be equal to the number of system states."
        assert set(u.data_vars) >= set(self.u_names), "All system inputs must be given."
        assert len(system_noise) == len(self.x), \
            "Length of system_noise must be equal to the number of system states."
        assert len(system_bias) == len(self.x), "Length of system_bias must be equal to the number of system states."
        assert len(measurement_noise) == len(self.z), \
            "Length of measurement_noise must be equal to the number of measurements."
        assert len(measurement_bias) == len(self.z), \
            "Length of measurement_bias must be equal to the number of measurements."

        # Make sure the noise variances and biases are in flattened ndarrays.
        system_noise = np.array(system_noise).flatten()
        system_bias = np.array(system_bias).flatten()
        measurement_noise = np.array(system_noise).flatten()
        measurement_bias = np.array(system_bias).flatten()

        # Get time vector
        time_vector = u.t.values
        u_log = np.vstack([u[u_name] for u_name in self.u_names]).T if len(self.u) else [[]] * len(time_vector)
        u_f_iter = iter(u_log)
        next(u_f_iter)
        n_x = len(self.x)
        n_z = len(self.z)

        # Create iterator that will give the final time of each iteration.
        t_f_iter = iter(time_vector)
        next(t_f_iter)

        # Create matrix in which to store results and set initial state
        x_log = np.empty((len(time_vector), len(self.x)))
        x_log.fill(np.nan)
        x_log[0, :] = x_0

        # Create measurement log
        z_log = np.empty((len(time_vector), len(self.x)))
        z_log.fill(np.nan)

        z_log[0, :] = self.h_func(0.0, *x_0, *u_log[0]).flatten() \
                      + measurement_noise * np.random.normal(size=(n_x,)) + measurement_bias

        # Iterate though each point in time and integrate using an euler integrator.
        for i, (t_i, t_f, x_i, u_i, u_f) in enumerate(zip(time_vector, t_f_iter, x_log, u_log, u_f_iter)):
            dt = t_f - t_i

            f = self.f_func(t_i, *x_i, *u_i).flatten()
            g = self.g_func(t_i, *x_i, *u_i)

            # Calculate derivative
            dx = f + (g @ (system_noise * np.random.normal(size=(n_x,)))).flatten() + system_bias
            x_f = x_i + dx * dt

            # Store result
            x_log[i+1, :] = x_f
            z_log[i+1, :] = self.h_func(t_f, *x_f, *u_f) + measurement_noise * np.random.normal(size=(n_x,)) + measurement_bias

        return u.merge(xr.Dataset(
            data_vars={
                **{self.x_names[x_idx]: (('t',), x_log[:, x_idx]) for x_idx in range(n_x)},
                **{self.z_names[z_idx]: (('t',), z_log[:, z_idx]) for z_idx in range(n_z)},
            }
        ))

    def filter(self,
               data: xr.Dataset,
               x_0: Union[np.ndarray, Sequence[float]],
               p_0: Union[np.ndarray, Sequence[Sequence[float]]],
               q: Union[np.ndarray, Sequence[Sequence[float]]],
               r: Union[np.ndarray, Sequence[Sequence[float]]],
               verbose=False):
        cols_set = set(data.data_vars)
        n_x = len(self.x)
        n_u = len(self.u)
        n_z = len(self.z)

        c_null = np.zeros((1, n_x))
        d_null = np.zeros((1, n_u))

        # Make sure p_0, Q and R are ndarray
        p_0 = np.asarray(p_0)
        q = np.asarray(q)
        r = np.asarray(r)

        m_g = self.g.shape[1]

        assert set(self.z_names) <= cols_set, f"Missing measurements in the data frame.\n{', '.join(set(self.z_names) - cols_set)}"
        assert len(x_0) == n_x, "x_0 must have the same size as the number of states."
        assert p_0.shape == (n_x, n_x), "p_0 must be a square matrix whose sides must have the same " \
                                        "length as the number of states."
        assert q.shape == (m_g, m_g), "The system noise covariance matrix 'Q' needs to be an nxn matrix, where n is" \
                                      "n the width of the G matrix."
        assert r.shape == (n_z, n_z), "The measurement noise covariance matrix 'R' needs to be an nxn matrix, where " \
                                      "n is the number of measurements."

        # Make sure x_0 is a flattened ndarray.
        x_0 = np.asarray(x_0).flatten()

        # Get time vector.
        time_vector = data.t.values

        # Get matrix with measurements and inputs.
        u_log = np.vstack([data[u_name].values for u_name in self.u_names]).T if len(self.u) else [[]] * len(time_vector)
        z_log = np.vstack([data[z_name].values for z_name in self.z_names]).T

        # Create iterator that will give the final time of each iteration.
        t_f_iter = iter(time_vector)
        next(t_f_iter)

        # Create logs
        x_k1k1_log = deque((x_0,), maxlen=len(time_vector)+2)
        p_k1k1_log = deque((p_0,), maxlen=len(time_vector)+2)
        iter_count = deque((0,), maxlen=len(time_vector)+2)
        # stdx_log = deque((p_0.diagonal().flatten(),), maxlen=len(time_vector)+2)
        # Constants
        eye_nx = np.eye(n_x)

        for k, (t_i, t_f, u_k, z_k) in tqdm(enumerate(zip(time_vector, t_f_iter, u_log, z_log)),
                                            total=len(time_vector)-1,
                                            disable=not verbose):
            # with profile():
            dt = t_f - t_i
            # print(dt)
            x_kk = x_k1k1_log[k]
            p_kk = p_k1k1_log[k]

            # State prediction
            x_kk1 = rk4(self.f_func, x_kk, u_k, t_i, t_f)

            k_gain = np.nan
            hx = np.nan
            p_kk1 = np.nan
            eta_i = np.nan
            eta_1i = x_kk1
            for i in range(self.max_iterations):
                # print(f"  {i}", eta_1i)
                # Measurement prediction
                z_p = self.h_func(t_i, *eta_1i, *u_k).flatten()

                # Calculate Phi and Gamma
                phi, gamma, _, __, ___ = cont2discrete(
                    (self.fx_func(t_i, *eta_1i, *u_k),
                     self.g_func(t_i, *eta_1i, *u_k),
                     c_null,
                     d_null,
                     ),
                    dt
                )

                # Covariance matrix prediction
                p_kk1 = phi @ p_kk @ phi.T + gamma @ q @ gamma.T

                # Calculate Kalman gain
                hx = self.hx_func(t_i, *eta_1i, *u_k)

                # Check observability.
                # o = obsv(fx, hx)
                # if np.linalg.matrix_rank(o) < n_x:
                #     print("Unobservable.")

                # Calculate Kalman gain.
                # Using pseudo inverse function since there might be non-invertible matrix.
                k_gain = (p_kk1 @ hx.T) @ np.linalg.pinv(hx @ p_kk1 @ hx.T + r)

                # Calculate optimal state
                eta_i = x_kk1 + k_gain @ (z_k - z_p - (hx @ (x_kk1 - eta_1i)))

                # Check if the error is small enough
                if (np.linalg.norm(eta_i - eta_1i) / np.linalg.norm(eta_1i)) <= self.eps:
                    # Break the loop if so.
                    x_k1k1 = eta_i.flatten()
                    iter_count.append(i+1)
                    break
                else:
                    # Set eta_1i and loop again.
                    eta_1i = eta_i.flatten()
            else:
                # We have run the maximum number of iterations. Set x_k1k1 with the current
                # value in eta_i
                x_k1k1 = eta_i
                iter_count.append(i+1)

            # p_k1k1 correction using numerically stable form of p_k1k1 = (eye(n) - K*hx) * p_kk_1
            D = (eye_nx - k_gain @ hx)
            p_k1k1 = D @ p_kk1 @ D.T + k_gain @ r @ k_gain.T

            # Store results
            x_k1k1_log.append(x_k1k1)
            p_k1k1_log.append(p_k1k1)
            # stdx_log.append(p_k1k1.diagonal().flatten().tolist())

        return data.merge(xr.Dataset(
            data_vars={
                **{f"{x_name}_est": (("t",), np.array(x_value)) for x_name, x_value in zip(self.x_names, (*zip(*x_k1k1_log),))},
                "p_k1k1": (("t", "dim_0", "dim_1"), np.array(p_k1k1_log)),
                'iekf_i_count': (("t",), np.array(iter_count)),
                # "stdx": (('t', "dim_0"), np.array(stdx_log))
            }
        ))


# def c2d(a: np.ndarray, b: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Calculates a discrete state space system from the continuous one.
#
#     :param a: Continuous A matrix
#     :param b: Continuous B matrix
#     :param dt: Time step
#     :return: Tuple with the discrete A and B matrices.
#     """
#     m, na = a.shape
#     m, nb = b.shape
#
#     s = expm(
#         np.vstack((
#             np.hstack((a, b)) * dt,
#             np.zeros((nb, na+nb))
#         ))
#     )
#
#     return s[:na, :na], s[:na, na:na+nb]


def rk4(f: Callable,
        x_0: Union[Sequence[float], np.ndarray],
        u: Union[Sequence[float], np.ndarray],
        t_i: float,
        t_f: float,
        n: int=1):
    """
    Fourth order Runge-Kutta integration.

    :param f: Callable returning the state derivatives. All parameters are scalars, the
              first parameter should be the time, followed by the current states and
              then the inputs.
    :param x_0: Sequence of initial state.
    :param u: Input vector
    :param t_i: Initial time
    :param t_f: Final time
    :param n: Number of iterations.
    :return: Final state vector.
    """
    w = np.asarray(x_0, dtype=np.float64).flatten()
    t = t_i
    h = (t_f - t_i) / n

    for j in range(1, n+1):
        k1 = h * f(t, *w, *u).flatten()
        k2 = h * f(t + h/2., *(w+k1/2.), *u).flatten()
        k3 = h * f(t + h/2., *(w+k2/2.), *u).flatten()
        k4 = h * f(t + h, *(w+k3), *u).flatten()

        w += (k1 + 2. * k2 + 2. * k3 + k4) / 6.0
        t = t_i + j * h

    return w

