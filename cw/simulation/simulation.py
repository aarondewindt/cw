from contextlib import contextmanager
from copy import deepcopy
from typing import Sequence, List, Set
from dataclasses import fields, is_dataclass

import numpy as np

from cw.simulation.integrator_base import IntegratorBase
from cw.simulation.states_base import StatesBase
from cw.simulation.module_base import ModuleBase
from cw.simulation.logging import LoggingBase
from cw.simulation.exception import SimulationError


class Simulation:
    def __init__(self, *,
                 states_class: type,
                 integrator: IntegratorBase,
                 modules: Sequence[ModuleBase],
                 logging: LoggingBase,
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
        self.discrete_states: Set[str] = set()
        self.continuous_modules: List[ModuleBase] = []

    @contextmanager
    def temporary_states(self, temporary=True, persist_discreet_states=False):
        if temporary:
            original_states = self.states
            self.states = deepcopy(original_states)
            yield
            if persist_discreet_states:
                # Copy the values from the discreet states into the original states.
                pass
            del self.states
            self.states = original_states
        else:
            yield

    def initialize(self):
        self.integrator.initialize(self)

        # Sort discrete and continuous modules
        for module in self.modules:
            if module.is_discreet:
                self.discrete_modules.append(module)
                self.discrete_states.update(module.output_states)
            else:
                self.continuous_modules.append(module)

            module.initialize(self)

        for field in fields(self.states):
            field_value = getattr(self.states, field.name)
            try:
                if np.isnan(field_value):
                    raise SimulationError(f"State {field.name} initial value contains 'Nan'.")

            except TypeError:
                if field_value is None:
                    raise SimulationError(f"State {field.name} initial value is 'None'.")

    def step_modules(self, run_discreet):
        for module in self.modules:
            if not module.is_discreet:
                module.step()
            else:
                if run_discreet:
                    module.step()

    def run(self, n_steps):
        self.integrator.run(n_steps)

