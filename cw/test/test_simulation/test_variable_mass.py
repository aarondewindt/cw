import numpy as np

import unittest
from dataclasses import dataclass
from pathlib import Path

from cw.simulation import StatesBase, Simulation, AB3Integrator, Logging, Plotter, ModuleBase
from cw.simulation.modules import VariableMass
from cw.flex_file import flex_load


class TestVariableMass(unittest.TestCase):
    def setUp(self) -> None:
        self.simulation = Simulation(
            states_class=VariableMassStates,
            integrator=AB3Integrator(
                h=0.01,
                rk4=True,
                fd_max_order=1),
            modules=[
                VariableMassTestModule(),
                VariableMass(
                    mass=(10, 5),
                    cg=([1, 0, 0], [0, 0, 0]),
                    inertia=(np.eye(3), np.zeros((3, 3))),
                ),
            ],
            logging=Logging(),
            initial_state_values=None,
        )

    def test_variable_mass(self):
        self.simulation.initialize()
        result = self.simulation.run(500)

        plotter = Plotter()
        plotter.plot_to_pdf(Path(__file__).parent / "variable_mass_results.i.pdf", result)


@dataclass
class VariableMassStates(StatesBase):
    t: float = 0
    mass: float = np.nan
    mass_dot: float = 0
    cg: np.ndarray = np.zeros(3)
    inertia_b: np.ndarray = np.zeros((3, 3))

    def get_y_dot(self):
        return self.mass_dot

    def get_y(self):
        return self.mass

    def set_t_y(self, t, y):
        self.t = t
        self.mass = y


class VariableMassTestModule(ModuleBase):
    def step(self):
        self.s.mass_dot = -1
