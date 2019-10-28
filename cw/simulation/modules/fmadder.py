from dataclasses import fields

import numpy as np

from cw.simulation.module_base import ModuleBase


class FMAdder(ModuleBase):
    def __init__(self):
        super().__init__(required_states=["force_b", "moment_b"])
        self.force_states = []
        self.moment_states = []

    def initialize(self, simulation):
        super().initialize(simulation)

        # List of all state names
        field_names = [field.name for field in fields(simulation.states)]

        # Find all input force and moment states
        for field_name in field_names:
            if field_name.startswith("force_") and (field_name != "force_b"):
                self.force_states.append(field_name)
            elif field_name.startswith("moment_") and (field_name != "moment_b"):
                self.moment_states.append(field_name)

    def step(self):
        total_force = np.zeros(3)
        total_moment = np.zeros(3)

        for force_name in self.force_states:
            total_force += self.s.__dict__[force_name]

        for moment_name in self.moment_states:
            total_moment += self.s.__dict__[moment_name]

        self.s.force_b = total_force
        self.s.moment_b = total_moment

