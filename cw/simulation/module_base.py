from typing import Optional, Sequence


class ModuleBase:
    def __init__(self, *,
                 clock_divider: Optional[int]=None,
                 target_time_step: Optional[float]=None,
                 is_discreet: bool=False,
                 output_states: Optional[Sequence[str]]=None
                 ):
        self.simulation = None
        self.clock_divider = clock_divider
        self.target_time_step = target_time_step
        self.is_discreet = is_discreet
        self.output_states = output_states

    def initialize(self, simulation):
        self.simulation = simulation

    def step(self):
        pass

    def end(self):
        pass
