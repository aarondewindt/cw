from cw.mp import NotebookProject
from typing import Iterator, NamedTuple


class ExampleNotebookProject(NotebookProject):
    # Batch input and output parameters.
    class InputTuple(NamedTuple):
        in_example_param_1: int
        in_example_param_2: float

    class OutputTuple(NamedTuple):
        out_example_param_1: bool
        out_example_param_2: float

    def __init__(self, path=None):
        super().__init__(path)

        self.batch_name = ""
        self.project_name = ""
        self.description = \
            """
            Description of the batch
            """

    def create_cases(self) -> Iterator[InputTuple]:
        """
        Either yield dictionaries with the input parameters for each case or return
        a list with all of these dictionaries.
        """
        return []

    def run_case(self, inputs: InputTuple) -> OutputTuple:
        """
        Function that takes in a dictionary with the input parameters
        for a single case and outputs

        :param inputs: Input parameters
        :return: Output parameters
        """
        raise NotImplemented()

