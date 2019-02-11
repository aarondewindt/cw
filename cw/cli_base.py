from abc import ABCMeta, abstractmethod
import argparse


class CLIBase:
    """
    Base class to give cli tools in cw a common interface to the main function.
    """
    @classmethod
    @abstractmethod
    def configure_arg_parser(cls, sub_parsers):
        """
        Used to create one or more subparsers. A default argument named func is required.
        The value of this argument must be the function called by the command and must
        take one parameter of type 'argparse.Namespace'.

        :param arg_parser: Subparsers object returned by 'ArgumentParser.add_subparsers()'
        """
        ...
