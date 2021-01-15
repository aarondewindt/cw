from typing import Optional, Callable, Deque, Tuple, Union
import numpy as np
from collections import deque
from math import remainder, fmod
import traceback

from cw.simulation.integrator_base import IntegratorBase
from cw.simulation.exception import SimulationError
from cw.simulation.differentiation import FiniteDifference


class AB3Integrator(IntegratorBase):
    def __init__(self, *,
                 h: Optional[float] = None,
                 rk4: bool,
                 fd_max_order: int=4):
        """
        Third order Adams–Bashforth integrator using a fourth order Runge–Kutta
        for the first four iterations.

        :param h: Integration step
        :param rk4: If `True` the RK4 methods will be used for the entire simulation.
        :param fd_max_order: Finite difference order used for differentiation.
        """
        super().__init__()
        self.rk4 = rk4
        self.fd = FiniteDifference(step_size=h, max_order=fd_max_order)
        self.h: float = h
        self.hdiv2: float = self.h / 2
        self.hdiv6: float = self.h / 6
        self.hdiv24: float = self.h / 24
        self.k: Deque[Union[None, float]] = deque([None, None, None, None])
        self.previous_step_y1 = None
        self.previous_step_t1 = None
        self.must_differentiate = None
        self.n_steps_since_reset = 0
        self.reset_states = False

    def initialize(self, simulation):
        self.simulation = simulation

    def run(self, n_steps: int):
        # Mark integrator as running
        self.running = True

        # We want to catch and print all exceptions except for the KeyboardInterrupt.
        try:
            # Check that the target time step of all discrete modules is integer
            # divisible by the integrator time step.
            # Set the module clock divider.
            for module in self.simulation.discrete_modules:
                if not (remainder(module.target_time_step, self.h) < 1e-8):
                    raise SimulationError(f"Discreet module '{module.__class__.__name__}' "
                                          f"target time step is not an integer multiple of the integrator time step.")
                else:
                    module.clock_divider = round(module.target_time_step / self.h)

            # Miscellaneous stuff that needs to be set before integrating.
            self.simulation.logging.reset(n_steps)
            t0 = self.simulation.states.t
            self.must_differentiate = self.simulation.states.get_differentiation_y() is not None

            # Previous step states.
            self.k: Deque[Union[None, float]] = deque([None, None, None, None])
            self.previous_step_t1 = self.simulation.states.t
            self.previous_step_y1 = self.simulation.states.get_y()

            # Reset the integrator, this clears the integrator history
            # and switches to the RK4 for the first few steps.
            # The `reset()` function can also be used by modules to
            # reset the history in case of a significant discrete event.
            self.reset()

            # Integrate
            last_step_idx = 0
            for step_idx in range(n_steps):
                last_step_idx = step_idx
                if not self.running:
                    break
                t = t0 + (step_idx + 1) * self.h
                self.run_single_step(step_idx, t, False)

            # Run one last step.
            last_step_idx += 1
            t = t0 + (last_step_idx + 1) * self.h
            self.run_single_step(last_step_idx, t, True)
        except KeyboardInterrupt:
            raise
        except:
            traceback.print_exc()
        finally:
            # Finish integration
            self.running = False
            for module in self.simulation.modules:
                module.run_end()
            return self.simulation.logging.finish()

    def run_single_step(self, step_idx: int, t1: float, is_last: bool):
        # Set the last step end state, as this step's start state.
        y0 = self.previous_step_y1
        t0 = self.previous_step_t1

        # Differentiate states if we have to.
        if self.must_differentiate:
            self.simulation.states.set_differentiation_y_dot(
                self.fd.differentiate(self.simulation.states.get_differentiation_y()))

        # Step all modules at t0.
        self.simulation.states.set_t_y(t0, y0)
        for module in self.simulation.modules:
            if module.is_discreet:
                if step_idx % module.clock_divider == 0:
                    module.run_step(is_last)
            else:
                module.run_step(is_last)

        if self.reset_states:
            self.reset_states = False
            y0 = self.simulation.states.get_y()

        # Log states at t0
        self.simulation.logging.log()

        # Derivative of y at t0.
        y0_dot = self.simulation.states.get_y_dot()

        # Use RK4 for the first few steps or if we are configured as an RK4 integrator.
        if (self.n_steps_since_reset < 4) or self.rk4:
            self.k.popleft()
            k1 = self.h * y0_dot
            k2 = self.h * self.simulation.get_y_dot(t0 + self.hdiv2, y0 + k1/2, is_last)
            k3 = self.h * self.simulation.get_y_dot(t0 + self.hdiv2, y0 + k2/2, is_last)
            k4 = self.h * self.simulation.get_y_dot(t0 + self.h, y0 + k3, is_last)
            y1 = y0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6

            # TODO: Possible bug, appending `h * y0_dot` here, while appending `y0_dot` in AB3 section
            self.k.append(k1)

        # Use Adams-Bashforth 3 otherwise.
        else:
            self.k.popleft()
            self.k.append(y0_dot)
            y1 = y0 + self.hdiv24 * (55 * self.k[3] - 59 * self.k[2] + 37 * self.k[1] - 9 * self.k[0])

        # Set the simulation state vector.
        self.simulation.states.set_t_y(t1, y1)

        # Store state information for the next step.
        self.previous_step_t1 = t1
        self.previous_step_y1 = y1
        self.n_steps_since_reset += 1

    def reset(self, states=False):
        self.n_steps_since_reset = 0
        if states:
            self.reset_states = True
