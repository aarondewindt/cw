import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import quaternion

from cw.simulation import StatesBase, Simulation, AB3Integrator, Logging, Plotter, ModuleBase
from cw.simulation.modules import EOM6DOF
from cw.flex_file import flex_load

from cw.conversions import dcm_to_q, q_to_euler


class TestEOM6DOF(unittest.TestCase):
    def setUp(self) -> None:
        self.verification_module = EOM6DOFVerificationDataModule()
        self.simulation = Simulation(
            states_class=EOM6DOFStates,
            integrator=AB3Integrator(
                h=0.001,
                rk4=True,
                fd_max_order=1),
            modules=[
                EOM6DOF(),
                self.verification_module
            ],
            logging=Logging(),
            initial_state_values=None,
        )

        verification_data_path = Path(__file__).parent / "eom6dof_verification_data.msgp.gz"
        self.verification_data = flex_load(verification_data_path)

    def test_eom6dof(self):
        self.simulation.initialize()
        self.verification_module.set_verification_data(self.verification_data[0])
        result = self.simulation.run(10000)

        state_names = ["ve", "xe", "dcm_eb", "vb", "omega_b", "omega_dot_b", "ab"]
        for state_name in state_names:
            result[f"error_{state_name}"] = (result[state_name] - result[f"correct_{state_name}"])
                                            # / result[state_name] * 100

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

    omega_b: np.ndarray = np.zeros(3)
    correct_omega_b: np.ndarray = np.zeros(3)

    omega_dot_b: np.ndarray = np.zeros(3)
    correct_omega_dot_b: np.ndarray = np.zeros(3)

    inertia: np.ndarray = np.eye(3)
    force: np.ndarray = np.zeros(3)
    inertia_dot: np.ndarray = np.zeros((3, 3))

    ab: np.ndarray = np.zeros(3)
    correct_ab: np.ndarray = np.zeros(3)

    vb: np.ndarray = np.zeros(3)
    correct_vb: np.ndarray = np.zeros(3)

    xe: np.ndarray = np.zeros(3)
    correct_xe: np.ndarray = np.zeros(3)

    q: np.quaternion = np.quaternion(1, 0, 0, 0)

    ve: np.ndarray = np.zeros(3)
    correct_ve: np.ndarray = np.zeros(3)

    moment: np.ndarray = np.zeros(3)

    dcm_eb: np.ndarray = quaternion.as_rotation_matrix(q)
    correct_dcm_eb: np.ndarray = quaternion.as_rotation_matrix(q)

    def get_y_dot(self):
        return np.hstack((self.omega_dot_b,
                          omega_to_q_dot(self.omega_b, self.q),
                          self.ab,
                          self.ve))

    def get_y(self):
        return np.hstack((
            self.omega_b,
            quaternion.as_float_array(self.q),
            self.vb,
            self.xe
        ))

    def set_t_y(self, t, y):
        self.t = t
        self.omega_b = y[:3]
        self.q = np.quaternion(*y[3:7])
        self.dcm_eb = quaternion.as_rotation_matrix(self.q)
        self.vb = y[7:10]
        self.ve = self.dcm_eb @ self.vb
        self.xe = y[10:]

    def get_differentiation_y(self):
        return self.inertia

    def set_differentiation_y_dot(self, y_dot):
        self.inertia_dot = y_dot


class EOM6DOFVerificationDataModule(ModuleBase):
    def __init__(self):
        super().__init__(required_states=["correct_ve", "correct_xe", "correct_dcm_eb", "correct_vb",
                                          "correct_omega_b", "correct_omega_dot_b", "correct_ab"],
                         is_discreet=True,
                         target_time_step=0.01)

        self.force = None
        self.moment = None
        self.mass = None
        self.inertia = None
        self.ve = None
        self.xe = None
        self.euler = None
        self.dcm_be = None
        self.vb = None
        self.omega_b = None
        self.omega_dot_b = None
        self.ab = None
        self.inertia_dot = None

    def set_verification_data(self, verification_data):
        self.force = np.array(verification_data["force"])
        self.moment = np.array(verification_data["moment"])
        self.mass = np.array(verification_data["mass"])
        self.inertia = np.array(verification_data["inertia"])
        self.ve = np.array(verification_data["ve"])
        self.xe = np.array(verification_data["xe"])
        self.euler = np.array(verification_data["euler"])
        self.dcm_be = np.array(verification_data["dcm_be"])
        self.vb = np.array(verification_data["vb"])
        self.omega_b = np.array(verification_data["omega_b"])
        self.omega_dot_b = np.array(verification_data["omega_dot_b"])
        self.ab = np.array(verification_data["ab"])
        self.inertia_dot = np.array(verification_data["inertia_dot"])
        ixx, iyy, izz, ixy, ixz, iyz = self.inertia[0, :]
        self.simulation.states.inertia = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])

    def initialize(self, simulation):
        super().initialize(simulation)

    def step(self):
        idx = round(self.s.t * 100)

        self.s.force = self.force[idx, :]
        self.s.moment = self.moment[idx, :]
        self.s.mass = self.mass[idx]
        ixx, iyy, izz, ixy, ixz, iyz = self.inertia[idx, :]
        self.s.inertia = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])

        self.s.correct_ve = self.ve[idx, :]
        self.s.correct_xe = self.xe[idx, :]
        self.s.correct_dcm_eb = self.dcm_be[..., idx].T
        self.s.correct_vb = self.vb[idx, :]
        self.s.correct_omega_b = self.omega_b[idx, :]
        self.s.correct_omega_dot_b = self.omega_dot_b[idx, :]
        self.s.correct_ab = self.ab[idx, :]
