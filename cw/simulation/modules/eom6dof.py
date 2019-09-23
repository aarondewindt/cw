from cw.simulation.module_base import ModuleBase
from dataclasses import fields


class EOM6DOF(ModuleBase):
    def __init__(self):
        super().__init__(required_states=[
            "force", "moment", "mass", "inertia", "inertia_dot",
            "ab", "vb", "ve", "xe",
            "omega_dot", "omega", "q"])

    def step(self):
        pass

