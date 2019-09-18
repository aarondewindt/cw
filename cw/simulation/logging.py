from collections import deque
from copy import deepcopy


class Logging:
    def __init__(self):
        self.simulation = None

        self.raw_log = None
        self.step_idx = 0

    def initialize(self, simulation):
        self.simulation = simulation

    def reset(self, n_steps):
        self.raw_log = deque([None]) * n_steps
        self.step_idx = 0

    def log(self):
        self.raw_log[self.step_idx] = deepcopy(self.simulation.states)
        self.step_idx += 1
