import numpy as np

from cw.simulation.module_base import ModuleBase


class EOM6DOF(ModuleBase):
    def __init__(self):
        super().__init__(required_states=[
            "force", "moment", "mass", "inertia", "inertia_dot",
            "ab", "vb", "ve", "xe",
            "omega_dot", "omega", "q"])

    def step(self):
        # Acceleration
        self.s.ab = self.s.force / self.s.mass + np.cross(self.s.vb, self.s.omega)

        # Rotational acceleration
        self.s.omega_dot = (self.s.moment
                            - self.s.inertia_dot @ self.s.omega
                            - np.cross(self.s.omega, self.s.inertia @ self.s.omega)
                            ).T @ np.linalg.inv(self.s.inertia)
