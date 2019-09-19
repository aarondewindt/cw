from typing import Optional, Sequence, Callable, Tuple
import numpy as np

from cw.simulation.exception import SimulationError


class IntegratorBase:
    def __init__(self):
        self.simulation = None

    def initialize(self, simulation):
        if self.simulation is None:
            raise SimulationError("Integrator not bound to simulation.")

    def run(self, n_steps):
        raise NotImplemented()

    def run_single_step(self, t1: float):
        raise NotImplemented()
