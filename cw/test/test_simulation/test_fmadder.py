import numpy as np

import unittest
from dataclasses import dataclass
from pathlib import Path

from cw.simulation import StatesBase, Simulation, AB3Integrator, Logging, Plotter, ModuleBase
from cw.simulation.modules import FMAdder
from cw.flex_file import flex_load


class TestEOM6DOF(unittest.TestCase):
    def setUp(self) -> None:
        self.simulation = Simulation(
            states_class=FMAdderStates,
            integrator=AB3Integrator(
                h=0.01,
                rk4=True,
                fd_max_order=1),
            modules=[
                FMAdderTestModule(),
                FMAdder(),
            ],
            logging=Logging(),
            initial_state_values=None,
        )

    def test_fmadder(self):
        self.simulation.initialize()
        result = self.simulation.run(3)

        force_sum = result.force_0 + result.force_1 + result.force_2 - result.force_b
        self.assertTrue(np.all(np.isclose(force_sum.values, 0.0)))

        moment_sum = result.moment_0 + result.moment_1 + result.moment_2 - result.moment_b
        self.assertTrue(np.all(np.isclose(moment_sum.values, 0.0)))


zero_array = np.zeros(1)


@dataclass
class FMAdderStates(StatesBase):
    t: float = 0
    
    force_b: np.ndarray = np.zeros(3)
    moment_b: np.ndarray = np.zeros(3)

    force_0: np.ndarray = np.zeros(3)
    moment_0: np.ndarray = np.zeros(3)

    force_1: np.ndarray = np.zeros(3)
    moment_1: np.ndarray = np.zeros(3)

    force_2: np.ndarray = np.zeros(3)
    moment_2: np.ndarray = np.zeros(3)

    def get_y_dot(self):
        return zero_array

    def get_y(self):
        return zero_array

    def set_t_y(self, t, y):
        self.t = t


class FMAdderTestModule(ModuleBase):
    def step(self):
        self.s.force_0 = np.random.random(3)
        self.s.moment_0 = np.random.random(3)
        self.s.force_1 = np.random.random(3)
        self.s.moment_1 = np.random.random(3)
        self.s.force_2 = np.random.random(3)
        self.s.moment_2 = np.random.random(3)
