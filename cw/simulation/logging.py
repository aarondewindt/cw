from collections import deque
from copy import deepcopy, copy
from dataclasses import fields

import numpy as np
import xarray as xr

from cw.itertools import until


class Logging:
    def __init__(self):
        self.simulation = None
        self.raw_log = None
        self.step_idx = 0
        self.results = None

    def initialize(self, simulation):
        self.simulation = simulation

    def reset(self, n_steps):
        self.results = None
        self.raw_log = [None] * n_steps
        self.step_idx = 0

    def log(self):
        self.raw_log[self.step_idx] = copy(self.simulation.states)
        self.step_idx += 1

    def finish(self):
        # Dictionary holding all field values.
        field_values = {field.name: ([], [None] * self.step_idx) for field in fields(self.raw_log[0])}

        for field_name, field_list in field_values.items():
            field_value = getattr(self.raw_log[0], field_name)
            field_list[0].append("t")
            if not np.isscalar(field_value):
                field_list[0].extend([f"d{i}" for i in range(np.ndim(field_value))])

        # Move all values to the dictionary
        for step_idx, step_data in enumerate(until(self.raw_log, None)):
            for field_name, field_list in field_values.items():
                field_list[1][step_idx] = getattr(step_data, field_name)

        # Time field
        t = field_values.pop("t")

        # Create data set and return it.
        self.results = xr.Dataset(field_values, coords={"t": t[1]})
        return self.results
