import unittest
from dataclasses import dataclass

import numpy as np

from cw.simulation import StatesBase, Simulation, AB3Integrator, Logging
from cw.simulation.modules import EOM6DOF


class TestEOM6DOF(unittest.TestCase):
    def setUp(self) -> None:
        self.simulation = Simulation(
            states_class=Sim1States,
            integrator=AB3Integrator(
                h=0.01,
                rk4=True,
                fd_max_order=1),
            modules=[
                EOM6DOF()
            ],
            logging=Logging(),
            initial_state_values=None,
        )

    def test_eom6dof(self):
        self.simulation.initialize()



@dataclass
class Sim1States(StatesBase):
    t: float = 0
    mass: float = 1
    omega: float = np.zeros(3)
    omega_dot: float = np.zeros(3)
    inertia: float =
    force: float =
    inertia_dot: float =
    ab: float =
    vb: float =
    xe: float =
    q: float =
    ve: float =
    moment: float =


    def get_y_dot(self):
        pass

    def get_y(self):
        pass

    def set_t_y(self, t, y):
        self.t = t
        self.s = y[0]
        self.v = y[1]

    def get_differentiation_y(self):
        pass

    def set_differentiation_y_dot(self, y_dot):
        pass