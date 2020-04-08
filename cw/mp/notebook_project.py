from cw.mp import BatchConfigurationBase, run_project_locally
from pathlib import Path
from typing import Iterator, NamedTuple, Optional
from textwrap import indent
import multiprocessing


class NotebookProject(BatchConfigurationBase):
    InputTuple: NamedTuple = None
    OutputTuple: NamedTuple = None

    def __init__(self, path=None):
        self.path = path or Path.cwd()
        self.project_path = path
        self.batch = self
        self.batch_name = ""
        self.project_name = ""
        self.description = ""

    @property
    def name(self):
        return self.batch_name

    @property
    def project(self):
        return self.project_name

    @staticmethod
    def validate_directory():
        return True

    @staticmethod
    def create_checksum_file(force=False):
        pass

    @staticmethod
    def print_example():
        with (Path(__file__).parent / "templates" / "notebook_configuration.py").open("r") as f:
            print(f.read())

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

    def info(self):
        input_param_str = indent('\n'.join(self.batch.input_parameters), " - ")
        output_param_str = indent('\n'.join(self.batch.output_parameters), " - ")
        print(f"path: {self.path}\n"
              f"project name: {self.project_name}\n"
              f"batch name: {self.batch_name}\n"
              f"number of cases: {self.batch.number_of_cases}\n"
              f"batch description:\n{indent(self.batch.description_trimmed, '   ')}\n"
              f"inputs: \n{input_param_str}\n"
              f"outputs: \n{output_param_str}"
              )

    def run_locally(self,
                    n_cores: int,
                    output_name: Optional[str] = None,
                    dump_interval: int = 5,
                    chunksize: int = 1,
                    verbose=False,
                    progress_bar=True):
        n_cores = int(n_cores or 0)
        max_cores = multiprocessing.cpu_count()
        n_cores = max_cores if (n_cores >= max_cores) or (n_cores < 0) else n_cores

        output_name = output_name or f"{self.batch_name}.i"
        return run_project_locally(self, output_name, n_cores, dump_interval, chunksize, verbose, False, progress_bar)
