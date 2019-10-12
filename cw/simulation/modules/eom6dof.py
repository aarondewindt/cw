import numpy as np
import quaternion
from numba import jit, f8

from cw.simulation.module_base import ModuleBase


class EOM6DOF(ModuleBase):
    def __init__(self):
        super().__init__(required_states=[
            "force", "moment", "mass", "inertia", "inertia_dot",
            "ab", "vb", "ve", "xe",
            "omega_dot_b", "omega_b", "q"])

    def step(self):
        # Acceleration
        # self.s.ab = self.s.force / self.s.mass + np.cross(self.s.vb, self.s.omega_b)
        #
        # # Rotational acceleration
        # self.s.omega_dot_b = (self.s.moment
        #                       - self.s.inertia_dot @ self.s.omega_b
        #                       - np.cross(self.s.omega_b, self.s.inertia @ self.s.omega_b)
        #                       ).T @ np.linalg.inv(self.s.inertia)

        self.s.ab, self.s.omega_dot_b = step_calc(self.s.force, self.s.moment, self.s.vb, self.s.omega_b, self.s.mass,
                                                  self.s.inertia, self.s.inertia_dot)


# @jit((f8[:], f8[:], f8[:], f8[:], f8, f8[:, :], f8[:, :]), nopython=True)
@jit(nopython=True)
def step_calc(force, moment, vb, omega_b, mass, inertia, inertia_dot):
    ab = force / mass + np.cross(vb, omega_b)

    omega_dot_b = (moment
                   - inertia_dot @ omega_b
                   - np.cross(omega_b, inertia @ omega_b)
                   ).T @ np.linalg.inv(inertia)

    return ab, omega_dot_b


def omega_to_q_dot(omega, quat, k=1.0):
    p, q, r = omega
    q_array = quaternion.as_float_array(quat)
    e = k * (1 - np.sum(q_array*q_array))
    return 0.5 * np.array([e * quat.w - p * quat.x - q * quat.y - r * quat.z,
                           p * quat.w + e * quat.x + r * quat.y - q * quat.z,
                           q * quat.w - r * quat.x + e * quat.y + p * quat.z,
                           r * quat.w + q * quat.x - p * quat.y + e * quat.z])


@jit(nopython=True)
def omega_to_q_dot_numba(omega, quat, k=1.0):
    p, q, r = omega
    e = k * (1 - np.sum(quat * quat))
    return 0.5 * np.array([e * quat[0] - p * quat[1] - q * quat[2] - r * quat[3],
                           p * quat[0] + e * quat[1] + r * quat[2] - q * quat[3],
                           q * quat[0] - r * quat[1] + e * quat[2] + p * quat[3],
                           r * quat[0] + q * quat[1] - p * quat[2] + e * quat[3]])
