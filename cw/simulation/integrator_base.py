from typing import Optional, Sequence, Callable, Tuple
import numpy as np

from cw.simulation.exception import SimulationError


class IntegratorBase:
    def __init__(self):
        self.simulation = None
        self.running = False

    def initialize(self, simulation):
        if self.simulation is None:
            raise SimulationError("Integrator not bound to simulation.")

    def run(self, n_steps):
        raise NotImplemented()

    def stop(self):
        self.running = False

    def run_single_step(self, step_idx: int, t1: float, last_iteration: bool):
        raise NotImplemented()

    def reset(self, states=False):
        raise NotImplemented()
