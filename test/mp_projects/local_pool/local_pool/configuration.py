from cw.mp import BatchConfigurationBase
from typing import Iterator, NamedTuple

from time import sleep
from itertools import product



class BatchConfiguration(BatchConfigurationBase):

    # Batch input and output parameters.
    class InputTuple(NamedTuple):
        in_a: float
        in_b: float

    class OutputTuple(NamedTuple):
        out_c: float
        out_d: float

    def __init__(self):
        super().__init__()

        self.name = "local_pool"  #: Batch name
        self.description = \
            """
            Description of the batch
            """
        self.project = ""  #: The name of the project this batch belongs to.

    def create_cases(self) -> Iterator[InputTuple]:
        """
        Either yield dictionaries with the input parameters for each case or return
        a list with all of these dictionaries.
        """
        for in_a, in_b in product(range(1), range(100)):
            yield self.InputTuple(in_a, in_b)

    def run_case(self, inputs: InputTuple) -> OutputTuple:
        """
        Function that takes in a dictionary with the input parameters
        for a single case and outputs

        :param inputs: Input parameters
        :return: Output parameters
        """
        out = self.OutputTuple(
            out_c=inputs.in_a + inputs.in_b,
            out_d=inputs.in_a * inputs.in_b
        )
        sleep(.01)
        return out
