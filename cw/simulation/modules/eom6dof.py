import numpy as np

from cw.simulation.module_base import ModuleBase


class EOM6DOF(ModuleBase):
    def __init__(self):
        super().__init__(required_states=[
            "force", "moment", "mass", "inertia", "inertia_dot",
            "ab", "vb", "ve", "xe",
            "omega_dot_b", "omega_b", "q"])

    def step(self):
        # Acceleration
        self.s.ab = self.s.force / self.s.mass + np.cross(self.s.vb, self.s.omega_b)

        # Rotational acceleration
        self.s.omega_dot_b = (self.s.moment
                              - self.s.inertia_dot @ self.s.omega_b
                              - np.cross(self.s.omega_b, self.s.inertia @ self.s.omega_b)
                              ).T @ np.linalg.inv(self.s.inertia)
