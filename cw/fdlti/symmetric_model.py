__author__ = 'Aaron M. de Windt'


from cw.cached import cached
import numpy as np
from control import StateSpace


class SymmetricModel(StateSpace):
    def __init__(self,
                 *,
                 v,
                 k_y_2,
                 c_bar,
                 mu_c,
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
                 c_m_delta_e
                 ):
        self.v = v
        self.k_y_2 = k_y_2
        self.c_bar = c_bar
        self.mu_c = mu_c
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

        super().__init__(self.__a_matrix, self.__b_matrix, self.__c_matrix, self.__d_matrix)

    @cached
    def __state_space_dynamics(self):
        """
        :return: Tuple with two numpy arrays with the A and B matrices of the
                 state space model.
        """

        vc = self.v / self.c_bar
        mu_c_2 = self.mu_c * 2
        mu_c_2_c_z_alpha_dot = mu_c_2 - self.c_z_alpha_dot
        mu_c_2_k_y_2 = mu_c_2 * self.k_y_2

        x_u = vc / mu_c_2 * self.c_x_u
        x_alpha = vc / mu_c_2 * self.c_x_alpha
        x_theta = vc / mu_c_2 * self.c_z_0
        x_q = vc / mu_c_2 * self.c_x_q
        x_delta_e = vc / mu_c_2 * self.c_x_delta_e

        z_u = vc / mu_c_2_c_z_alpha_dot * self.c_z_u
        z_alpha = vc / mu_c_2_c_z_alpha_dot * self.c_z_alpha
        z_theta = -vc / mu_c_2_c_z_alpha_dot * self.c_x_0
        z_q = vc / mu_c_2_c_z_alpha_dot * (mu_c_2 + self.c_z_q)
        z_delta_e = vc / mu_c_2_c_z_alpha_dot * self.c_z_delta_e

        m_u = vc / mu_c_2_k_y_2 * (self.c_m_u + self.c_z_u * self.c_m_alpha_dot / mu_c_2_c_z_alpha_dot)
        m_alpha = vc / mu_c_2_k_y_2 * (self.c_m_alpha + self.c_z_alpha * self.c_m_alpha_dot / mu_c_2_c_z_alpha_dot)
        m_theta = -vc / mu_c_2_k_y_2 * self.c_x_0 * self.c_m_alpha_dot / mu_c_2_c_z_alpha_dot
        m_q = vc / mu_c_2_k_y_2 * (self.c_m_q + self.c_m_alpha_dot * (2 * self.mu_c + self.c_z_q) / mu_c_2_c_z_alpha_dot)
        m_delta_e = vc / mu_c_2_k_y_2 * (self.c_m_delta_e + self.c_z_delta_e * self.c_m_alpha_dot / mu_c_2_c_z_alpha_dot)

        a_matrix = np.array([
            [x_u, x_alpha, x_theta, x_q],
            [z_u, z_alpha, z_theta, z_q],
            [  0,       0,       0,  vc],
            [m_u, m_alpha, m_theta, m_q]])

        b_matrix = np.array([[x_delta_e], [z_delta_e], [0], [m_delta_e]])

        return a_matrix, b_matrix

    @cached
    def __a_matrix(self):
        return self.__state_space_dynamics[0]

    @cached
    def __b_matrix(self):
        return self.__state_space_dynamics[1]

    @cached
    def __c_matrix(self):
        return np.eye(4)

    @cached
    def __d_matrix(self):
        return np.zeros((4, 1))
