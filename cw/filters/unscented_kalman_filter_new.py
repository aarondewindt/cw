import numpy as np
import xarray as xr
# import sympy as sp
import control as ct

from typing import Callable, Optional, Sequence, Union, Tuple
from collections import deque
from textwrap import indent
from tqdm.auto import tqdm
from scipy.linalg import expm
from IPython.display import display, Markdown
from scipy.signal import cont2discrete
import html2text

from cw.unscented_transform import UnscentedTransform
from cw.vdom import div, h3, css, dl, dt, dd, b, pre


class UnscentedKalmanFilter:
    def __init__(self, *,
                 x_names: Sequence[str],
                 u_names: Sequence[str],
                 z_names: Sequence[str],
                 f: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
                 f_alpha=1,
                 f_beta=2,
                 f_k=None,
                 h: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
                 h_alpha=1,
                 h_beta=2,
                 h_k=None,
                 g: Callable[[float, np.ndarray, np.ndarray], np.ndarray]):
        # Functions
        self.f = f
        self.h = h
        self.g = g

        self.x_names = tuple(x_names)
        self.u_names = tuple(u_names)
        self.z_names = tuple(z_names)

        # Unscented transformation parameters
        self.f_alpha = f_alpha
        self.f_beta = f_beta
        self.f_k = 3 - len(self.x_names) if f_k is None else f_k
        self.h_k = 3 - len(self.x_names) if h_k is None else h_k
        self.h_alpha = h_alpha
        self.h_beta = h_beta

    def __repr__(self):
        return html2text.html2text(self._repr_html_())

    def _repr_html_(self):
        return div(
            h3("UnscentedKalmanFilter"),
            div(css(margin_left="2em"),
                dl(
                    dt(b("x_names")), dd(", ".join(self.x_names)),
                    dt(b("z_names")), dd(", ".join(self.z_names)),

                    dt(b("f_alpha")), dd(self.f_alpha),
                    dt(b("f_beta")), dd(self.f_beta),
                    dt(b("f_k")), dd(self.f_k),

                    dt(b("h_alpha")), dd(self.h_alpha),
                    dt(b("h_beta")), dd(self.h_beta),
                    dt(b("h_k")), dd(self.h_k),
                ))
        ).to_html()

    def sim(self,
            *,
            data: xr.Dataset,
            system_noise: Sequence[float],
            system_bias: Sequence[float],
            measurement_noise: Sequence[float],
            measurement_bias: Sequence[float],
            inplace=True) -> xr.Dataset:

        if not inplace:
            data = data.copy()

        initial_conditions = []
        inputs = []
        states_idxs = []
        inputs_idxs = []
        for idx, variable_name in enumerate(self.x_names):
            if variable_name in data:
                variable: xr.DataArray = data[variable_name]
                if variable.dims == ("t",):
                    inputs.append(variable.values)
                    inputs_idxs.append(idx)
                elif variable.dims == ():
                    initial_conditions.append(variable.item())
                    states_idxs.append(idx)
            else:
                raise ValueError("Dataset must contain an initial condition or full state values for all states.")

        initial_conditions = np.array(initial_conditions)
        inputs = np.array(inputs).T
        states_idxs = states_idxs
        inputs_idxs = inputs_idxs

        n_x = len(self.x_names)
        n_z = len(self.z_names)

        # Check inputs.
        assert len(system_noise) == n_x, \
            "Length of system_noise must be equal to the number of system states."
        assert len(system_bias) == n_x, "Length of system_bias must be equal to the number of system states."
        assert len(measurement_noise) == n_z, \
            "Length of measurement_noise must be equal to the number of measurements."
        assert len(measurement_bias) == n_z, \
            "Length of measurement_bias must be equal to the number of measurements."

        # Make sure the noise variances and biases are in flattened ndarrays.
        system_noise = np.asarray(system_noise).flatten()
        system_bias = np.asarray(system_bias).flatten()
        measurement_noise = np.asarray(measurement_noise).flatten()
        measurement_bias = np.asarray(measurement_bias).flatten()

        # Get time vector
        time_vector = data.t.values
        n_t = len(time_vector)

        # Iterator that gives the final input for each iteration.
        inputs_final_iterator = iter(inputs)
        next(inputs_final_iterator)

        # Create iterator that will give the final time of each iteration.
        time_final_iter = iter(time_vector)
        next(time_final_iter)

        # Create matrix in which to store results and set initial state
        x_log = np.empty((n_t, n_x))
        x_log.fill(np.nan)

        # Create measurement log
        z_log = np.empty((n_t, n_z))
        z_log.fill(np.nan)

        # Set initial values
        x_log[0, states_idxs] = initial_conditions
        x_log[0, inputs_idxs] = inputs[0, :]

        z_log[0, :] = self.h(0.0, x_log[0, :]).flatten() \
                      + measurement_noise * np.random.normal(size=(n_z,)) + measurement_bias

        # Iterate though each point in time and integrate using an euler integrator.
        for i, (t_i, t_f, x_i, u_i, u_f) in enumerate(zip(time_vector, time_final_iter,
                                                          x_log,
                                                          inputs, inputs_final_iterator)):
            x_i = x_log[i, :]

            f = self.f(t_i, x_i).flatten()
            g = self.g(t_i, x_i)

            # Calculate derivative
            dx = f + (g @ (system_noise * np.random.normal(size=(n_x,)))).flatten() + system_bias
            x_f = x_i + dx * (t_f - t_i)

            x_f[inputs_idxs] = u_f

            # Store result
            x_log[i+1, :] = x_f
            z_log[i+1, :] = \
                self.h(t_f, x_f).flatten() \
                + measurement_noise * np.random.normal(size=(n_z,)) + measurement_bias

        return data.merge(xr.Dataset(
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
        n_x = len(self.x_names)
        n_z = len(self.z_names)
        n_u = len(self.u_names)

        # Make sure p_0, Q and R are ndarray
        p_0 = np.asarray(p_0)
        q = np.asarray(q)
        r = np.asarray(r)

        assert set(self.z_names) <= cols_set, f"Missing measurements in the data frame.\n{', '.join(set(self.z_names) - cols_set)}"
        assert len(x_0) == n_x, "x_0 must have the same size as the number of states."
        assert p_0.shape == (n_x, n_x), "p_0 must be a square matrix whose sides must have the same " \
                                        "length as the number of states."
        assert r.shape == (n_z, n_z), "The measurement noise covariance matrix 'R' needs to be an nxn matrix, where " \
                                      "n is the number of measurements."

        # Make sure x_0 is a flattened ndarray.
        x_0 = np.asarray(x_0).flatten()

        # Get time vector.
        time_vector = data.t.values

        # Get matrix with measurements and inputs.
        u_log = np.vstack([data[u_name].values for u_name in self.u_names]).T if n_u else [[]] * len(time_vector)
        z_log = np.vstack([data[z_name].values for z_name in self.z_names]).T

        # Create iterator that will give the final time of each iteration.
        t_f_iter = iter(time_vector)
        next(t_f_iter)

        # Create logs
        x_k1k1_log = deque((x_0,), maxlen=len(time_vector) + 2)
        p_k1k1_log = deque((p_0,), maxlen=len(time_vector) + 2)
        x_kk1_log = deque((x_0,), maxlen=len(time_vector) + 2)
        p_kk1_log = deque((p_0,), maxlen=len(time_vector) + 2)

        t_i = np.nan
        t_f = np.nan
        u_k = np.nan

        def f_eval(x):
            return rk4(self.f, x.flatten(), u_k, t_i, t_f)

        def h_eval(x):
            return self.h(t_i, x.flatten(), u_k)

        f_ut = UnscentedTransform(f_eval, self.f_alpha, self.f_beta, self.f_k)
        h_ut = UnscentedTransform(h_eval, self.h_alpha, self.h_beta, self.h_k)

        for k, (t_i, t_f, u_k, z_k) in tqdm(enumerate(zip(time_vector, t_f_iter, u_log, z_log)),
                                            total=len(time_vector)-1,
                                            disable=not verbose):
            x_kk = x_k1k1_log[k]
            p_kk = p_k1k1_log[k]

            # State and covariance matrix prediction
            x_kk1, p_kk1, _ = f_ut(x_kk, p_kk)
            p_kk1 += q

            # Update
            z_p, s_k, c_k = h_ut(x_kk1, p_kk1)
            s_k += r

            # Calculate Kalman gain
            k_gain = c_k @ np.linalg.pinv(s_k)
            x_k1k1 = x_kk1 - (k_gain @ (z_k - z_p))
            p_k1k1 = p_kk1 - (k_gain @ (s_k @ k_gain.T))

            # Store results
            x_k1k1_log.append(x_k1k1)
            p_k1k1_log.append(p_k1k1)
            x_kk1_log.append(x_kk1)
            p_kk1_log.append(p_kk1)

        # Add results to the dataset and return it
        data["p_k1k1"] = (("t", "dim_0", "dim_1"), np.array(p_k1k1_log))
        data["x_k1k1"] = (("t", "x_idx"), np.array(x_k1k1_log))
        data["p_kk1"] = (("t", "dim_0", "dim_1"), np.array(p_kk1_log))
        data["x_kk1"] = (("t", "x_idx"), np.array(x_kk1_log))

        for x_idx, x_name in enumerate(self.x_names):
            data[f"{x_name}_filtered"] = data.x_k1k1.isel(x_idx=x_idx)
            data[f"{x_name}_filtered_std"] = data.p_k1k1.isel(dim_0=x_idx, dim_1=x_idx)

        return data


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
        k1 = h * f(t, w, u).flatten()
        k2 = h * f(t + h/2., (w+k1/2.), u).flatten()
        k3 = h * f(t + h/2., (w+k2/2.), u).flatten()
        k4 = h * f(t + h, (w+k3), u).flatten()

        w += (k1 + 2. * k2 + 2. * k3 + k4) / 6.0
        t = t_i + j * h

    return w
