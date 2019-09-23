from typing import Optional, Sequence


class ModuleBase:
    def __init__(self, *,
                 target_time_step: Optional[float]=None,
                 is_discreet: bool=False):
        if is_discreet:
            if target_time_step is None:
                raise ValueError("target_time_step must be set for discreet modules.")
        self.simulation = None
        self.target_time_step = target_time_step
        self.is_discreet = is_discreet
        self.s = None

    def initialize(self, simulation):
        self.simulation = simulation

    def run_step(self):
        self.s = self.simulation.states
        self.step()
        del self.s

    def step(self):
        pass

    def end(self):
        pass
