__author__ = 'Aaron M. de Windt'


from .symmetric_model import SymmetricModel
from .asymmetric_model import AsymmetricModel

from cw.cached import cached
from cw.atmo import alt_from_rho

import numpy as np
import control as ct
import matplotlib.pyplot as plt


# template = FlightLTI(
#     s=,         k_x_2=,
#     m=,         k_y2=,
#     l_h=,       k_z_2=,
#     x_cg=,      k_xz=,
#     v=,         c_l=,
#     c_bar=,     mu_c=,
#     b=,         mu_b=,
#
#     c_x_0=,          c_z_0=,
#     c_x_u=,          c_z_u=,          c_m_u=,
#     c_x_alpha=,      c_z_alpha=,      c_m_alpha=,
#     c_x_alpha_dot=,  c_z_alpha_dot=,  c_m_alpha_dot=,
#     c_x_q=,          c_z_q=,          c_m_q=,
#     c_x_delta_e=,    c_z_delta_e=,    c_m_delta_e=,
#
#     c_y_beta=,       c_l_beta=,       c_n_beta=,
#     c_y_p=,          c_l_p=,          c_n_p=,
#     c_y_r=,          c_l_r=,          c_n_r=,
#     c_y_delta_a=,    c_l_delta_a=,    c_n_delta_a=,
#     c_y_delta_r=,    c_l_delta_r=,    c_n_delta_r=,
# )


class FlightLTI:
    def __init__(self,
                 *,
                 name,
                 description,
                 s,
                 m,
                 l_h,
                 x_cg,
                 v,
                 c_bar,
                 b,
                 mu_c,
                 mu_b,
                 k_x_2,
                 k_y_2,
                 k_z_2,
                 k_xz,
                 c_l,
                 c_x_0,
                 c_x_u,
                 c_x_alpha,
                 c_x_alpha_dot,
                 c_x_q,
                 c_x_delta_e,
                 c_z_0,
                 c_z_u,
                 c_z_alpha,
                 c_z_alpha_dot,
                 c_z_q,
                 c_z_delta_e,
                 c_m_u,
                 c_m_alpha,
                 c_m_alpha_dot,
                 c_m_q,
                 c_m_delta_e,
                 c_y_beta,
                 c_y_p,
                 c_y_r,
                 c_y_delta_a,
                 c_y_delta_r,
                 c_l_beta,
                 c_l_p,
                 c_l_r,
                 c_l_delta_a,
                 c_l_delta_r,
                 c_n_beta,
                 c_n_p,
                 c_n_r,
                 c_n_delta_a,
                 c_n_delta_r
                 ):

        self.name = name
        self.description = description
        self.s = s
        self.m = m
        self.l_h = l_h
        self.x_cg = x_cg
        self.v = v
        self.c_bar = c_bar
        self.b = b
        self.mu_c = mu_c
        self.mu_b = mu_b
        self.k_x_2 = k_x_2
        self.k_y_2 = k_y_2
        self.k_z_2 = k_z_2
        self.k_xz = k_xz
        self.c_l = c_l
        self.c_x_0 = c_x_0
        self.c_x_u = c_x_u
        self.c_x_alpha = c_x_alpha
        self.c_x_alpha_dot = c_x_alpha_dot
        self.c_x_q = c_x_q
        self.c_x_delta_e = c_x_delta_e
        self.c_z_0 = c_z_0
        self.c_z_u = c_z_u
        self.c_z_alpha = c_z_alpha
        self.c_z_alpha_dot = c_z_alpha_dot
        self.c_z_q = c_z_q
        self.c_z_delta_e = c_z_delta_e
        self.c_m_u = c_m_u
        self.c_m_alpha = c_m_alpha
        self.c_m_alpha_dot = c_m_alpha_dot
        self.c_m_q = c_m_q
        self.c_m_delta_e = c_m_delta_e
        self.c_y_beta = c_y_beta
        self.c_y_p = c_y_p
        self.c_y_r = c_y_r
        self.c_y_delta_a = c_y_delta_a
        self.c_y_delta_r = c_y_delta_r
        self.c_l_beta = c_l_beta
        self.c_l_p = c_l_p
        self.c_l_r = c_l_r
        self.c_l_delta_a = c_l_delta_a
        self.c_l_delta_r = c_l_delta_r
        self.c_n_beta = c_n_beta
        self.c_n_p = c_n_p
        self.c_n_r = c_n_r
        self.c_n_delta_a = c_n_delta_a
        self.c_n_delta_r = c_n_delta_r

        self.symmetric = SymmetricModel(
            v=v,
            k_y_2=k_y_2,
            c_bar=c_bar,
            mu_c=mu_c,
            c_x_0=c_x_0,
            c_x_u=c_x_u,
            c_x_alpha=c_x_alpha,
            c_x_alpha_dot=c_x_alpha_dot,
            c_x_q=c_x_q,
            c_x_delta_e=c_x_delta_e,
            c_z_0=c_z_0,
            c_z_u=c_z_u,
            c_z_alpha=c_z_alpha,
            c_z_alpha_dot=c_z_alpha_dot,
            c_z_q=c_z_q,
            c_z_delta_e=c_z_delta_e,
            c_m_u=c_m_u,
            c_m_alpha=c_m_alpha,
            c_m_alpha_dot=c_m_alpha_dot,
            c_m_q=c_m_q,
            c_m_delta_e=c_m_delta_e,
        )

        self.asymmetric = AsymmetricModel(
            v=v,
            b=b,
            k_x_2=k_x_2,
            k_z_2=k_z_2,
            k_xz=k_xz,
            c_l=c_l,
            mu_b=mu_b,
            c_y_beta=c_y_beta,
            c_y_p=c_y_p,
            c_y_r=c_y_r,
            c_y_delta_a=c_y_delta_a,
            c_y_delta_r=c_y_delta_r,
            c_l_beta=c_l_beta,
            c_l_p=c_l_p,
            c_l_r=c_l_r,
            c_l_delta_a=c_l_delta_a,
            c_l_delta_r=c_l_delta_r,
            c_n_beta=c_n_beta,
            c_n_p=c_n_p,
            c_n_r=c_n_r,
            c_n_delta_a=c_n_delta_a,
            c_n_delta_r=c_n_delta_r,
        )

    @cached
    def rho(self):
        return self.m / (self.mu_b * self.s * self.b)

    @cached
    def alt(self):
        return alt_from_rho(self.rho)

    @cached
    def i_xx(self):
        return self.k_x_2 * self.m * self.b * self.b

    @cached
    def i_yy(self):
        return self.k_y_2 * self.m * self.c_bar * self.c_bar

    @cached
    def i_zz(self):
        return self.k_z_2 * self.m * self.b * self.b

    @cached
    def i_xz(self):
        return self.k_xz * self.m * self.b * self.b

    def sim_elevator_step(self, magnitude=-0.005, end_time=100):
        """
        Simulates an elevator step input of -0.005 [rad].

        :returns: tuple (time, y_out)
            WHERE
            np.ndarray time is the time vector.
            np.ndarray y_out is the symmetric flight dynamics LTI output vector. Contains the results
              for Velocity, aoa, flight path angle and pitch rate.
        """
        t = np.arange(0, end_time, 0.01)
        u = np.ones(t.shape) * magnitude
        t, y_out, _ = ct.forced_response(self.symmetric, t, u)
        y_out[0, :] *= self.v
        y_out[0, :] += self.v
        y_out[3, :] *= self.v / self.c_bar
        return {
            "t": t,
            "u": y_out[0, :],
            "alpha": y_out[1, :],
            "theta": y_out[2, :],
            "q": y_out[3, :],
            "delta_e": u,
        }

    # def fig_elevator_step(self, magnitude=-0.005, end_time=100):
    #     """
    #     :return: A matplotlib figure showing the results of an elevator step input.
    #     """
    #     t, y_out, u = self.sim_elevator_step(magnitude, end_time)
    #
    #     fig = plt.figure(figsize=(10, 9))
    #     plt.suptitle("Elevator step response")
    #
    #     ax = plt.subplot(321)
    #     plt.plot(t, y_out[0])
    #     ax.set_ylabel(r"$V\ [m/s]$")
    #
    #     ax = plt.subplot(322)
    #     plt.plot(t, y_out[1])
    #     ax.set_ylabel(r"$\alpha\ [rad]$")
    #
    #     ax = plt.subplot(323)
    #     plt.plot(t, y_out[2])
    #     ax.set_ylabel(r"$\theta\ [rad]$")
    #     ax.set_xlabel("$t [s]$")
    #
    #     ax = plt.subplot(324)
    #     plt.plot(t, y_out[3])
    #     ax.set_ylabel(r"$q\ [rad]$")
    #     ax.set_xlabel("$t\ [s]$")
    #
    #     ax = plt.subplot(313)
    #     plt.plot(t, u)
    #     ax.set_ylabel(r"$u\ [rad]$")
    #     ax.set_xlabel("$t\ [s]$")
    #
    #     fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    #
    #     return fig

    def sim_rudder_pulse(self, magnitude=0.025, end_time=15):
        t = np.arange(0, end_time, 0.01)
        u = np.zeros((2, t.shape[0]))
        u[1, np.where(t <= 1)] = magnitude

        t, y_out, _ = ct.forced_response(self.asymmetric, t, u)
        y_out[2, :] *= 2 * self.v / self.b
        y_out[3, :] *= 2 * self.v / self.b
        return {
            "t": t,
            "beta": y_out[0, :],
            "phi": y_out[1, :],
            "p": y_out[2, :],
            "r": y_out[3, :],
            "delta_a": u[0, :],
            "delta_r": u[1, :],
        }

    # def fig_rudder_pulse(self, magnitude=0.025, end_time=15):
    #     """
    #     :return: A matplotlib figure showing the results of a rudder step input.
    #     """
    #     t, y_out, u = self.sim_rudder_pulse(magnitude, end_time)
    #
    #     fig = plt.figure(figsize=(10, 9))
    #     plt.suptitle("Rudder step response")
    #
    #     ax = plt.subplot(321)
    #     plt.plot(t, y_out[0])
    #     ax.set_ylabel(r"$\beta\ [rad]$")
    #
    #     ax = plt.subplot(322)
    #     plt.plot(t, y_out[1])
    #     ax.set_ylabel(r"$\phi\ [rad]$")
    #
    #     ax = plt.subplot(323)
    #     plt.plot(t, y_out[2])
    #     ax.set_ylabel(r"$p\ [rad/s]$")
    #     ax.set_xlabel("$t\ [s]$")
    #
    #     ax = plt.subplot(324)
    #     plt.plot(t, y_out[3])
    #     ax.set_ylabel(r"$r\ [rad/s]$")
    #     ax.set_xlabel("$t\ [s]$")
    #
    #     ax = plt.subplot(313)
    #     plt.plot(t, u[1, :])
    #     ax.set_ylabel(r"$u\ [rad]$")
    #     ax.set_xlabel("$t\ [s]$")
    #
    #     fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    #
    #     return fig

    def sim_aileron_pulse(self, magnitude=0.025, end_time=15):
        t = np.arange(0, end_time, 0.01)
        u = np.zeros((2, t.shape[0]))
        u[0, np.where(t <= 1)] = magnitude

        t, y_out, _ = ct.forced_response(self.asymmetric, t, u)
        y_out[2, :] *= 2 * self.v / self.b
        y_out[3, :] *= 2 * self.v / self.b
        return {
            "t": t,
            "beta": y_out[0, :],
            "phi": y_out[1, :],
            "p": y_out[2, :],
            "r": y_out[3, :],
            "delta_a": u[0, :],
            "delta_r": u[1, :],
        }

    # def fig_aileron_pulse(self, magnitude=0.025, end_time=15):
    #     """
    #     :return: A matplotlib figure showing the results of an aileron step input.
    #     """
    #     t, y_out, u = self.sim_aileron_pulse(magnitude, end_time)
    #
    #     fig = plt.figure(figsize=(10, 9))
    #     plt.suptitle("Aileron step response")
    #
    #     ax = plt.subplot(321)
    #     plt.plot(t, y_out[0])
    #     ax.set_ylabel(r"$\beta\ [rad]$")
    #
    #     ax = plt.subplot(322)
    #     plt.plot(t, y_out[1])
    #     ax.set_ylabel(r"$\phi\ [rad]$")
    #
    #     ax = plt.subplot(323)
    #     plt.plot(t, y_out[2])
    #     ax.set_ylabel(r"$p\ [rad/s]$")
    #     ax.set_xlabel("$t\ [s]$")
    #
    #     ax = plt.subplot(324)
    #     plt.plot(t, y_out[3])
    #     ax.set_ylabel(r"$r\ [rad/s]$")
    #     ax.set_xlabel("$t\ [s]$")
    #
    #     ax = plt.subplot(313)
    #     plt.plot(t, u[0, :])
    #     ax.set_ylabel(r"$u\ [rad]$")
    #     ax.set_xlabel("$t\ [s]$")
    #
    #     fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    #
    #     return fig