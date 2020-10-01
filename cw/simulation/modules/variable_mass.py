from typing import Tuple, Sequence
from math import isclose

import numpy as np

from cw.simulation.module_base import ModuleBase
from cw.simulation.exception import SimulationError


class VariableMass(ModuleBase):
    def __init__(self,
                 mass: Tuple[float, float],
                 cg: Tuple[Sequence, Sequence],
                 inertia: Tuple[Sequence, Sequence],
                 ):
        super().__init__(required_states=[
            "mass",
            "inertia_b", "cg"])

        self.mass_0 = mass[0]
        self.mass_1 = mass[1]
        self.cg_0 = np.asarray(cg[0])
        self.cg_1 = np.asarray(cg[1])
        self.inertia_0 = np.asarray(inertia[0])
        self.inertia_1 = np.asarray(inertia[1])

        # Difference between max and min values
        self.d_mass = self.mass_1 - self.mass_0
        self.d_cg = self.cg_1 - self.cg_0
        self.d_inertia = self.inertia_1 - self.inertia_0

        self.is_mass_constant = isclose(self.d_mass, 0.0, abs_tol=1e-8)

    def initialize(self, simulation):
        super().initialize(simulation)
        # Set initial values.
        simulation.states.mass = self.mass_0
        simulation.states.cg = self.cg_0
        simulation.states.inertia_b = self.inertia_0

    def step(self):
        if self.is_mass_constant:
            self.s.cg = self.cg_0
            self.s.inertia_b = self.inertia_0
        else:
            if self.mass_0 >= self.s.mass >= self.mass_1:
                # Interpolate w.r.t. mass to get the cg and inertia.
                k = (self.s.mass - self.mass_0) / self.d_mass
                self.s.cg = self.cg_0 + self.d_cg * k
                self.s.inertia_b = self.inertia_0 + self.d_inertia * k
            else:
                raise SimulationError("Mass out of range.")
