import numpy as np

from cw.simulation import Simulation, StatesBase, AB3Integrator, ModuleBase

from dataclasses import dataclass

nan = float('nan')


def main():
    simulation = Simulation(
        states_class=Sim1States,
        integrator=AB3Integrator(
            h=1,
            rk4=False),
        modules=[
            ModuleA()
        ],
        logging=None,
        initial_state_values=None,
    )

    simulation.initialize()

    simulation.run(10)


@dataclass
class Sim1States(StatesBase):
    t: float = 0
    s: float = 0
    v: float = 0
    a: float = 0

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
        simulation.states.a = 1

    def step(self):
        self.simulation.states.a = 1

main()
