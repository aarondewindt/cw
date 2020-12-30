import numpy as np
import numba as nb
from numba.experimental import jitclass


spec = (
    ("k_p", nb.float64),
    ("k_i", nb.float64),
    ("k_d", nb.float64),
    ("initialized", nb.boolean),
    ("integral", nb.float64),
    ("derivative", nb.float64),
    ("error", nb.float64),
    ("time", nb.float64),
    ("output", nb.float64),
)


@jitclass(spec)
class PIDScalarNumba:
    def __init__(self, k_p, k_i, k_d):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d

        self.initialized = False
        self.integral = 0.0
        self.derivative = 0.0
        self.error = np.nan
        self.time = np.nan
        self.output = np.nan

    def reset(self):
        if not self.initialized:
            return

        self.initialized = False
        self.integral = 0.0
        self.derivative = 0.0
        self.error = np.nan
        self.time = np.nan
        self.output = np.nan

    def step(self, time, command, value):
        if not self.initialized:
            # For the first iteration we run a P controller and set the initial values.
            self.initialized = True
            self.error = command - value
            self.time = time
            self.output = self.k_p * self.error

        else:
            error = command - value
            dt = time - self.time
            self.time = time

            self.integral += (self.error + error) / 2 * dt
            self.derivative = (error - self.error) / dt
            self.error = error

            self.output = error * self.k_p + self.integral * self.k_i + self.derivative * self.k_d

        return self.output


