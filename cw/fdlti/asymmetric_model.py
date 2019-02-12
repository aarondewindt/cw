__author__ = 'Aaron M. de Windt'

from cw.cached import cached
import numpy as np
from control import StateSpace


class AsymmetricModel(StateSpace):
    def __init__(self,
                 *,
                 v,
                 b,
                 k_x_2,
                 k_z_2,
                 k_xz,
                 c_l,
                 mu_b,
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
        self.v = v
        self.b = b
        self.k_x_2 = k_x_2
        self.k_z_2 = k_z_2
        self.k_xz = k_xz
        self.c_l = c_l
        self.mu_b = mu_b
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

        self.c_y_beta_dot = 0
        self.c_n_beta_dot = 0

        super().__init__(self.__a_matrix, self.__b_matrix, self.__c_matrix, self.__d_matrix)

    @cached
    def __state_space_dynamics(self):
        """
        :return: Tuple with two numpy arrays with the A and B matrices of the
                 state space model.
        """

        # return self.__state_space_dynamics_alternative

        y_factor = self.v / (2 * self.b * self.mu_b)
        l_n_factor = self.v / (4 * self.b * self.mu_b * (self.k_x_2 * self.k_z_2 - self.k_xz**2))

        y_beta =    y_factor * self.c_y_beta
        y_phi =     y_factor * self.c_l
        y_p =       y_factor * self.c_y_p
        y_r =       y_factor * (self.c_y_r - 4 * self.mu_b)
        y_delta_a = y_factor * self.c_y_delta_a
        y_delta_r = y_factor * self.c_y_delta_r

        l_beta =    l_n_factor * (self.c_l_beta    * self.k_z_2 + self.c_n_beta    * self.k_xz)
        l_p =       l_n_factor * (self.c_l_p       * self.k_z_2 + self.c_n_p       * self.k_xz)
        l_r =       l_n_factor * (self.c_l_r       * self.k_z_2 + self.c_n_r       * self.k_xz)
        l_delta_a = l_n_factor * (self.c_l_delta_a * self.k_z_2 + self.c_n_delta_a * self.k_xz)
        l_delta_r = l_n_factor * (self.c_l_delta_r * self.k_z_2 + self.c_n_delta_r * self.k_xz)

        n_beta =    l_n_factor * (self.c_l_beta    * self.k_xz  + self.c_n_beta    * self.k_x_2)
        n_p =       l_n_factor * (self.c_l_p       * self.k_xz  + self.c_n_p       * self.k_x_2)
        n_r =       l_n_factor * (self.c_l_r       * self.k_xz  + self.c_n_r       * self.k_x_2)
        n_delta_a = l_n_factor * (self.c_l_delta_a * self.k_xz  + self.c_n_delta_a * self.k_x_2)
        n_delta_r = l_n_factor * (self.c_l_delta_r * self.k_xz  + self.c_n_delta_r * self.k_x_2)

        l_phi = 0
        n_phi = 0
        phi_p = 2 * self.v / self.b

        a_matrix = [
            [y_beta, y_phi,   y_p, y_r],
            [     0,     0, phi_p,   0],
            [l_beta, l_phi,   l_p, l_r],
            [n_beta, n_phi,   n_p, n_r],
        ]

        b_matrix = [
            [y_delta_a, y_delta_r],
            [        0,         0],
            [l_delta_a, l_delta_r],
            [n_delta_a, n_delta_r]
        ]

        return a_matrix, b_matrix

    @cached
    def __state_space_dynamics_alternative(self):
        C1 = np.zeros((4, 4))
        C2 = np.zeros((4, 4))
        C3 = np.zeros((4, 2))

        C1_1_1 = (self.c_y_beta_dot - 2 * self.mu_b) * self.b / self.v
        C1_2_2 = -0.5 * self.b / self.v
        C1_3_3 = -4 * self.mu_b * self.k_x_2 * self.b * self.b / 2 / self.v ** 2
        C1_3_4 = 4 * self.mu_b * self.k_xz * self.b * self.b / 2 / self.v ** 2
        C1_4_1 = self.c_n_beta_dot * self.b / self.v
        C1_4_3 = 4 * self.mu_b * self.k_xz * self.b * self.b / 2 / self.v ** 2
        C1_4_4 = -4 * self.mu_b * self.k_z_2 * self.b * self.b / 2 / self.v ** 2

        C2_1_1 = self.c_y_beta
        C2_1_2 = self.c_l
        C2_1_3 = self.b / 2 / self.v * self.c_y_p
        C2_1_4 = self.b / 2 / self.v * (self.c_y_r - 4 * self.mu_b)
        C2_2_3 = self.b / 2 / self.v
        C2_3_1 = self.c_l_beta
        C2_3_3 = self.b / 2 / self.v * self.c_l_p
        C2_3_4 = self.b / 2 / self.v * self.c_l_r
        C2_4_1 = self.c_n_beta
        C2_4_3 = self.b / 2 / self.v * self.c_n_p
        C2_4_4 = self.b / 2 / self.v * self.c_n_r

        C3_1_1 = self.c_y_delta_a
        C3_1_2 = self.c_y_delta_r
        C3_3_1 = self.c_l_delta_a
        C3_3_2 = self.c_l_delta_r
        C3_4_1 = self.c_n_delta_a
        C3_4_2 = self.c_n_r

        C1 = np.array([
            [C1_1_1, 0, 0, 0],
            [0, C1_2_2, 0, 0],
            [0, 0, C1_3_3, C1_3_4],
            [C1_4_1, 0, C1_4_3, C1_4_4]])

        C2 = np.array([
            [C2_1_1, C2_1_2, C2_1_3, C2_1_4],
            [0, 0, C2_2_3, 0],
            [C2_3_1, 0, C2_3_3, C2_3_4],
            [C2_4_1, 0, C2_4_3, C2_4_4]])

        C3 = np.array([
            [C3_1_1, C3_1_2],
            [     0,      0],
            [C3_3_1, C3_3_2],
            [C3_4_1, C3_4_2],
        ])

        C1_inv = np.linalg.inv(C1)
        a_matrix = -C1_inv @ C2
        b_matrix = -C1_inv @ C3

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
        return np.zeros((4, 2))
