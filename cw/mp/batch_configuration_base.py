from abc import abstractmethod, ABCMeta
import sys
from typing import NamedTuple, Iterator

from cw.cached import cached_class, cached


class BatchConfigurationBase(metaclass=ABCMeta):
    """
    Abstract class used to define a new batch.
    """

    InputTuple: NamedTuple = None
    OutputTuple: NamedTuple = None

    def __init__(self):
        self.name = None  #: Batch name
        self.description = None  #: Batch description.
        self.project = None  #: The name of the project this batch belongs to.

    @cached_class
    def input_parameters(cls):
        return cls.InputTuple._fields

    @cached_class
    def output_parameters(cls):
        return cls.OutputTuple._fields

    @property
    def description_trimmed(self):
        text = self.description

        if not text:
            return ''
        # Convert tabs to spaces (following the normal Python rules)
        # and split into a list of lines:
        lines = text.expandtabs().splitlines()
        # Determine minimum indentation (first line doesn't count):
        indent = sys.maxsize
        for line in lines[1:]:
            stripped = line.lstrip()
            if stripped:
                indent = min(indent, len(line) - len(stripped))
        # Remove indentation (first line is special):
        trimmed = [lines[0].strip()]
        if indent < sys.maxsize:
            for line in lines[1:]:
                trimmed.append(line[indent:].rstrip())
        # Strip off trailing and leading blank lines:
        while trimmed and not trimmed[-1]:
            trimmed.pop()
        while trimmed and not trimmed[0]:
            trimmed.pop(0)
        # Return a single string:
        return '\n'.join(trimmed)

    @cached
    def number_of_cases(self) -> int:
        """
        (Cached) property returning the number of cases in the batch. If the number
        of cases cannot be known before the cases are generated, return None.
        """

        # Notes about this default implementation.
        #  - Assumes a finite number of cases.
        #  - Inefficient since it generates the entire list of cases.
        #  - Assumes that the number of cases will not change between calls.

        # It is recommended you calculate the total number of cases for your analysis
        # and return that.

        return sum(1 for x in self.create_cases())

    @abstractmethod
    def create_cases(self) -> Iterator[InputTuple]:
        """
        Abstract generator that yields the input parameters for each case.

        :return:
        """
        pass

    @abstractmethod
    def run_case(self, inputs: InputTuple) -> OutputTuple:
        """
        Abstract function that takes in a dictionary with the input parameters
        for a single case and outputs

        :param inputs: Input parameters
        :return: Output parameters
        """
        raise NotImplemented()


