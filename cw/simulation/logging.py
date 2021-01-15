from collections import deque
from copy import deepcopy, copy
from dataclasses import fields
from abc import ABC, abstractmethod

import numpy as np
import xarray as xr

from cw.itertools import until


class LoggerBase(ABC):
    @abstractmethod
    def initialize(self, simulation):
        pass

    @abstractmethod
    def reset(self, n_steps):
        pass

    @abstractmethod
    def log(self):
        pass

    @abstractmethod
    def finish(self):
        pass


class Logging(LoggerBase):
    def __init__(self,
                 sample_divider: int=1):
        self.simulation = None
        self.raw_log = None
        self.step_idx = 0
        self.results = None
        self.sample_divider = sample_divider
        self.logged_samples = 0

    def initialize(self, simulation):
        self.simulation = simulation

    def reset(self, n_steps):
        self.results = None
        self.raw_log = [None] * n_steps
        self.step_idx = 0
        self.logged_samples = 0

    def log(self):
        if not (self.step_idx % self.sample_divider):
            self.raw_log[self.logged_samples] = copy(self.simulation.states)
            self.logged_samples += 1
        self.step_idx += 1

    def finish(self):
        if self.logged_samples:
            # Dictionary holding all field values.
            field_values = {field.name: ([], [None] * self.logged_samples) for field in fields(self.simulation.states_class)}

            for field_name, field_list in field_values.items():
                field_value = getattr(self.raw_log[0], field_name)
                field_list[0].append("t")
                if not np.isscalar(field_value):
                    field_list[0].extend([f"d_{field_value.shape[i]}_{i}" for i in range(np.ndim(field_value))])

            # Move all values to the dictionary
            for step_idx, step_data in enumerate(until(self.raw_log, None)):
                for field_name, field_list in field_values.items():
                    field_list[1][step_idx] = getattr(step_data, field_name)

            # Time field
            t = field_values.pop("t")

            attributes = {}
            for module in self.simulation.modules:
                if module_attributes := module.get_attributes():
                    attributes.update(module_attributes)

            # Create data set and return it.
            self.results = xr.Dataset(field_values, coords={"t": t[1]}, attrs=attributes)
            return self.results
        else:
            return None


class LastValueLogger(LoggerBase):
    def __init__(self):
        self.last_state = None
        self.simulation = None

    def initialize(self, simulation):
        self.simulation = simulation

    def reset(self, n_steps):
        self.last_state = None

    def log(self):
        pass

    def finish(self):
        return copy(self.simulation.states)


class BatchLogger(LoggerBase):
    def __init__(self):
        self.last_results = None
        self.simulation = None
        self.raw_log = deque()
        self.results = None

        self.log_full_episode = False

        self.episode_raw_log = deque()

    def initialize(self, simulation):
        self.simulation = simulation

    def reset(self, n_steps):
        self.episode_raw_log = deque()

    def reset_batch(self):
        self.raw_log = deque()

    def log(self):
        if self.log_full_episode:
            self.episode_raw_log.append(copy(self.simulation.states))

    def finish(self):
        result = copy(self.simulation.states)
        self.raw_log.append(result)

        if self.log_full_episode and self.episode_raw_log:
            # Dictionary holding all field values.
            field_values = {field.name: ([], [None] * len(self.episode_raw_log)) for field in
                            fields(self.simulation.states_class)}

            for field_name, field_list in field_values.items():
                field_value = getattr(self.episode_raw_log[0], field_name)
                field_list[0].append("t")
                if not np.isscalar(field_value):
                    field_list[0].extend([f"d_{field_value.shape[i]}_{i}" for i in range(np.ndim(field_value))])

            # Move all values to the dictionary
            for step_idx, step_data in enumerate(until(self.episode_raw_log, None)):
                for field_name, field_list in field_values.items():
                    field_list[1][step_idx] = getattr(step_data, field_name)

            # Time field
            t = field_values.pop("t")

            attributes = {}
            for module in self.simulation.modules:
                if module_attributes := module.get_attributes():
                    attributes.update(module_attributes)

            # Create data set and return it.
            return xr.Dataset(field_values, coords={"t": t[1]}, attrs=attributes)

        return result

    def finish_batch(self):
        if self.raw_log:
            # Dictionary holding all field values.
            field_values = {field.name: ([], [None] * len(self.raw_log)) for field in fields(self.simulation.states_class)}

            for field_name, field_list in field_values.items():
                field_value = getattr(self.raw_log[0], field_name)
                field_list[0].append("idx")
                if not np.isscalar(field_value):
                    field_list[0].extend([f"d_{field_value.shape[i]}_{i}" for i in range(np.ndim(field_value))])

            # Move all values to the dictionary
            for step_idx, step_data in enumerate(until(self.raw_log, None)):
                for field_name, field_list in field_values.items():
                    field_list[1][step_idx] = getattr(step_data, field_name)

            attributes = {}
            for module in self.simulation.modules:
                if module_attributes := module.get_attributes():
                    attributes.update(module_attributes)

            # Create data set and return it.
            return xr.Dataset(field_values, coords={"idx": range(len(self.raw_log))}, attrs=attributes)
        else:
            return None
