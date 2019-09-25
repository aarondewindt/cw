import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import quaternion

from cw.simulation import StatesBase, Simulation, AB3Integrator, Logging, Plotter, ModuleBase
from cw.simulation.modules import EOM6DOF
from cw.cached import cached


class TestEOM6DOF(unittest.TestCase):
    def setUp(self) -> None:
        self.simulation = Simulation(
            states_class=EOM6DOFStates,
            integrator=AB3Integrator(
                h=0.01,
                rk4=True,
                fd_max_order=1),
            modules=[
                EOM6DOF(),
                TestModule()
            ],
            logging=Logging(),
            initial_state_values=None,
        )

    def test_eom6dof(self):
        self.simulation.initialize()
        result = self.simulation.run(1000)
        plotter = Plotter()
        plotter.plot_to_pdf(Path(__file__).parent / "eom6dof_results.i.pdf", result)


def omega_to_q_dot(omega, quat, k=1.0):
    p, q, r = omega
    q_array = quaternion.as_float_array(quat)
    e = k * (1 - sum(q_array*q_array))
    return 0.5 * np.array([e * quat.w - p * quat.x - q * quat.y - r * quat.z,
                           p * quat.w + e * quat.x + r * quat.y - q * quat.z,
                           q * quat.w - r * quat.x + e * quat.y + p * quat.z,
                           r * quat.w + q * quat.x - p * quat.y + e * quat.z])


@dataclass
class EOM6DOFStates(StatesBase):
    t: float = 0
    mass: float = 1
    omega: np.ndarray = np.zeros(3)
    omega_dot: np.ndarray = np.zeros(3)
    inertia: np.ndarray = np.eye(3)
    force: np.ndarray = np.zeros(3)
    inertia_dot: np.ndarray = np.zeros((3, 3))
    ab: np.ndarray = np.zeros(3)
    vb: np.ndarray = np.zeros(3)
    xe: np.ndarray = np.zeros(3)
    q: np.quaternion = np.quaternion(1, 0, 0, 0)
    ve: np.ndarray = np.zeros(3)
    moment: np.ndarray = np.zeros(3)

    dcm_be = quaternion.as_rotation_matrix(q)

    def get_y_dot(self):
        return np.hstack((self.omega_dot,
                          omega_to_q_dot(self.omega, self.q),
                          self.ab,
                          self.ve))

    def get_y(self):
        return np.hstack((
            self.omega,
            quaternion.as_float_array(self.q),
            self.vb,
            self.xe
        ))

    def set_t_y(self, t, y):
        self.t = t
        self.omega = y[:3]
        self.q = np.quaternion(*y[3:7])
        self.dcm_be = quaternion.as_rotation_matrix(self.q)
        self.vb = y[7:10]
        self.ve = self.dcm_be.T @ self.vb
        self.xe = y[10:]

    def get_differentiation_y(self):
        return self.inertia

    def set_differentiation_y_dot(self, y_dot):
        self.inertia_dot = y_dot


class TestModule(ModuleBase):
    def step(self):
        self.s.force = np.array([1, 0, 0])

