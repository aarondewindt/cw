from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from cw.simulation import Simulation, StatesBase, AB3Integrator, ModuleBase, Logging

nan = float('nan')


def main():
    simulation = Simulation(
        states_class=Sim1States,
        integrator=AB3Integrator(
            h=0.001,
            rk4=True),
        modules=[
            ModuleA()
        ],
        logging=Logging(),
        initial_state_values=None,
    )

    simulation.initialize()

    result = simulation.run(10000)

    # print(result)
    #
    # plt.figure()
    # result.s[:, 0].plot()
    # result.s[:, 1].plot()
    # result.s[:, 2].plot()
    #
    # plt.show()


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
        return np.array([self.v, self.a])

    def get_y(self):
        return np.array([self.s, self.v])

    def set_t_y(self, t, y):
        self.t = t
        self.s = y[0]
        self.v = y[1]


class ModuleA(ModuleBase):
    def __init__(self):
        super().__init__()

    def initialize(self, simulation):
        super().initialize(simulation)
        simulation.states.a = np.array([1, 2, 3])

    def step(self):
        pass
        # self.simulation.states.a = 1

main()
