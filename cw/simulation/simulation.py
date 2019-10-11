from contextlib import contextmanager
from copy import deepcopy, copy
from typing import Sequence, List, Set
from dataclasses import fields, is_dataclass
from math import fmod

import numpy as np

from cw.simulation.integrator_base import IntegratorBase
from cw.simulation.states_base import StatesBase
from cw.simulation.module_base import ModuleBase
from cw.simulation.logging import Logging
from cw.simulation.exception import SimulationError


class Simulation:
    def __init__(self, *,
                 states_class: type,
                 integrator: IntegratorBase,
                 modules: Sequence[ModuleBase],
                 logging: Logging,
                 initial_state_values=None):
        self.integrator = integrator
        self.modules = modules
        self.logging = logging
        if is_dataclass:
            self.states_class = states_class
        else:
            raise TypeError("The states class must be a dataclass.")
        self.states: StatesBase = states_class(**(initial_state_values or {}))

        self.discrete_modules: List[ModuleBase] = []
        self.continuous_modules: List[ModuleBase] = []

    @contextmanager
    def temporary_states(self, temporary=True):
        if temporary:
            original_states = self.states
            self.states = copy(original_states)
            yield
            del self.states
            self.states = original_states
        else:
            yield

    def initialize(self):
        self.integrator.initialize(self)
        self.logging.initialize(self)

        # Sort discrete and continuous modules
        # and check for required states
        for module in self.modules:
            if module.is_discreet:
                self.discrete_modules.append(module)
            else:
                self.continuous_modules.append(module)

            if module.required_states:
                states_set = set(module.required_states)
                for field in fields(self.states):
                    if field.name in states_set:
                        states_set.remove(field.name)
                if len(states_set):
                    raise SimulationError(
                        f"Missing required states for module '{module.name}': {', '.join(states_set)}")

            module.initialize(self)

        time_field_valid = False
        for field in fields(self.states):
            field_value = getattr(self.states, field.name)
            try:
                if np.any(np.isnan(field_value)):
                    raise SimulationError(f"State {field.name} initial value contains 'Nan'.")
            except TypeError:
                if field_value is None:
                    raise SimulationError(f"State {field.name} initial value is 'None'.")

            if field.name == "t":
                time_field_valid = field.type == float

        if not time_field_valid:
            raise SimulationError("The time field 't' in the states dataclass is missing or of invalid type.")

    def run(self, n_steps):
        return self.integrator.run(n_steps)

    def get_y_dot(self, t, y, *, temporary=True):
        """
        Executes all of the continuous modules and returns the state vector derivative.
        """
        with self.temporary_states(temporary):
            # Set the simulation state.
            self.states.set_t_y(t, y)
            # Run continuous modules
            for module in self.continuous_modules:
                module.run_step()
            # Get the state vector derivative.
            return self.states.get_y_dot()

    def step_continuous_modules(self, t, y):
        """
        Executes all of the continuous modules.
        """
        self.states.set_t_y(t, y)
        for module in self.continuous_modules:
            module.run_step()

    def step_discrete_modules(self):
        for module in self.discrete_modules:
            module.run_step()

    # def step_all_modules(self, t, y):
    #     self.states.set_t_y(t, y)
    #     for module in self.modules:
    #         if module.is_discreet:
    #             if fmod(t, module.target_time_step) < 1e-10:
    #                 module.run_step()
    #         else:
    #             module.run_step()
