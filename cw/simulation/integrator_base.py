from typing import Optional, Sequence, Callable, Tuple
import numpy as np

from cw.simulation.exception import SimulationError


class IntegratorBase:
    def __init__(self):
        self.simulation = None

    def initialize(self, simulation):
        if self.simulation is None:
            raise SimulationError("Integrator not bound to simulation.")

    def get_y_dot(self, t, y, *, run_discreet=False, temporary=True):
        with self.simulation.temporary_states(temporary, persist_discreet_states=run_discreet):
            self.simulation.states.set_t_y(t, y)
            self.simulation.step_modules(run_discreet)
            return self.simulation.states.get_y_dot()

    def run_at_t_y(self, t, y):
        self.simulation.states.set_t_y(t, y)
        self.simulation.step_modules(False)

    def run(self, n_steps):
        raise NotImplemented()

    def run_single_step(self, t1: float):
        raise NotImplemented()
