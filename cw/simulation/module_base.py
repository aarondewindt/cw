from abc import ABC, abstractmethod
from typing import Optional, Sequence


class ModuleBase(ABC):
    def __init__(self, *,
                 target_time_step: Optional[float]=None,
                 is_discreet: bool=False,
                 required_states=None):
        if is_discreet:
            if target_time_step is None:
                raise ValueError("target_time_step must be set for discreet modules.")
        self.simulation = None
        self.target_time_step = target_time_step
        self.is_discreet = is_discreet
        self.required_states = required_states
        self.s = None

    @property
    def name(self):
        return self.__class__.__name__

    def initialize(self, simulation):
        self.simulation = simulation

    def run_step(self, is_last):
        self.s = self.simulation.states
        self.step(is_last)
        del self.s

    def run_end(self):
        self.s = self.simulation.states
        self.end()
        del self.s

    def step(self, is_last):
        pass

    def end(self):
        pass

    def get_attributes(self):
        pass
