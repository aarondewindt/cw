from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from cw.simulation import Simulation, StatesBase, AB3Integrator, ModuleBase, Logging, Plotter
from cw.context import time_it


nan = float('nan')


def main():
    simulation = Simulation(
        states_class=Sim1States,
        integrator=AB3Integrator(
            h=0.001,
            rk4=True),
        modules=[
            ModuleA(),
            ModuleB()
        ],
        logging=Logging(),
        initial_state_values=None,
    )

    simulation.initialize()

    with time_it("simulation run"):
        result = simulation.run(10)

    plotter = Plotter()
    plotter.process_results(result)
    plt.show()


@dataclass
class Sim1States(StatesBase):
    t: float = 0
    mass: float = 10
    s: np.ndarray = np.zeros(3)
    v: np.ndarray = np.zeros(3)
    a: np.ndarray = np.zeros(3)
    # i: np.ndarray = np.eye(3)
    state: str = "qwerty"

    def get_y_dot(self):
        return np.array([self.v, self.a], dtype=np.float)

    def get_y(self):
        return np.array([self.s, self.v], dtype=np.float)

    def set_t_y(self, t, y):
        self.t = t
        self.s = y[0]
        self.v = y[1]


class ModuleA(ModuleBase):
    def __init__(self):
        super().__init__()

    def initialize(self, simulation):
        super().initialize(simulation)
        simulation.states.a = np.array([0., 0., 0.])

    def step(self):
        print("Module A step")
        # self.simulation.states.a = 1


class ModuleB(ModuleBase):
    def __init__(self):
        super().__init__(is_discreet=True)

    def initialize(self, simulation):
        super().initialize(simulation)

    def step(self):
        print("Module B step")
        a = self.simulation.states.a[0]
        self.simulation.states.a = np.array([1, 0, 0])
main()
