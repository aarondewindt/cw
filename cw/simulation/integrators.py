from typing import Optional, Callable, Deque, Tuple, Union
import numpy as np
from collections import deque

from cw.simulation.integrator_base import IntegratorBase
from cw.simulation.exception import SimulationError


class AB3Integrator(IntegratorBase):
    def __init__(self, *,
                 h: Optional[float] = None,
                 rk4: bool):
        super().__init__()
        self.rk4 = rk4
        self.h: float = h
        self.hdiv2: float = self.h / 2
        self.hdiv6: float = self.h / 6
        self.hdiv24: float = self.h / 24
        self.k: Deque[Union[None, float]] = deque([None, None, None, None])
        self.previous_step_y1 = None
        self.previous_step_t1 = None

    def initialize(self, simulation):
        self.simulation = simulation

    def run(self, n_steps: int):
        self.simulation.logging.reset(n_steps)
        t = self.simulation.states.t
        for step_idx in range(n_steps):
            t += self.h
            self.run_single_step(t)
        return self.simulation.logging.finish()

    def run_single_step(self, t1: float):
        # Get the time and state vector.
        # If this is our first iteration then we need to get them from the
        # states object.
        if self.previous_step_y1 is None:
            y0 = self.simulation.states.get_y()
            t0 = self.simulation.states.t
        else:
            y0 = self.previous_step_y1
            t0 = self.previous_step_t1

        # Step all modules at t0.
        self.simulation.step_all_modules(t0, y0)

        # Log states at t0
        self.simulation.logging.log()

        # Derivative of y at t0.
        y0_dot = self.simulation.states.get_y_dot()

        if not np.isclose(t1, t0 + self.h, rtol=1e-09, atol=1e-11):
            raise SimulationError("Only fixed step size allowed. Expected: " + str(self.h) + ", received: " + str(t1-t0))

        # Use RK4 for the first few steps or if we have to use it.
        if (self.k[1] is None) or self.rk4:
            self.k.pop()
            k1 = self.h * y0_dot
            k2 = self.h * self.simulation.get_y_dot(t0 + self.hdiv2, y0 + k1/2)
            k3 = self.h * self.simulation.get_y_dot(t0 + self.hdiv2, y0 + k2/2)
            k4 = self.h * self.simulation.get_y_dot(t0 + self.h, y0 + k3)
            y1 = y0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            self.k.append(k1)

        # Use Adams-Bashforth 3 when possible
        else:
            self.k.pop()
            self.k.append(y0_dot)
            y1 = y0 + self.hdiv24 * (55 * self.k[3] - 59 * self.k[2] + 37 * self.k[1] - 9 * self.k[0])

        self.simulation.states.set_t_y(t1, y1)
        self.previous_step_t1 = t1
        self.previous_step_y1 = y1
