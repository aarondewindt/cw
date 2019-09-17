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
        self.hdiv2: float = self.h/2
        self.hdiv6: float = self.h / 6
        self.hdiv24: float = self.h / 24
        self.k: Deque[Union[None, float]] = deque([None, None, None, None])

    def initialize(self, simulation):
        self.simulation = simulation

    def run(self, n_steps: int):
        t = self.simulation.states.t
        for step_idx in range(n_steps):
            t += self.h
            self.run_single_step(t)

            print(self.simulation.states)

    def run_single_step(self, t1: float):
        # Get the time and state vector
        y0 = self.simulation.states.get_y()
        t0 = self.simulation.states.t

        if t1 == t0:
            return y0, t1

        elif not np.isclose(t1, t0 + self.h, rtol=1e-09, atol=1e-11):
            raise SimulationError("Only fixed step size allowed. Expected: " + str(self.h) + ", received: " + str(t1-t0))
        else:
            pass
        # Use RK4 for the first few steps or if we have to use it.
        if (self.k[1] is None) or self.rk4:
            self.k.pop()

            k1 = self.h * self.get_y_dot(t0, y0, run_discreet=True)
            k2 = self.h * self.get_y_dot(t0 + self.hdiv2, y0 + 0.5 * k1)
            k3 = self.h * self.get_y_dot(t0 + self.hdiv2, y0 + 0.5 * k2)
            k4 = self.h * self.get_y_dot(t0 + self.h, y0 + k3)

            t = t0 + self.h
            y1 = y0 + self.hdiv6 * (k1 + 2 * k2 + 2 * k3 + k4)
            self.k.append(k1)

        # Use Adams-Bashforth 3 when possible
        else:
            self.k.pop()
            self.k.append(self.get_y_dot(t0, y0, run_discreet=True))
            t = t0 + self.h
            y1 = y0 + self.hdiv24 * (55 * self.k[3] - 59 * self.k[2] + 37 * self.k[1] - 9 * self.k[0])

        # Set the final state vector and rerun all of the modules to update the rest of the states.
        # Except the discrete states.
        self.run_at_t_y(t, y1)
