import re
from io import StringIO
from cw.cli_base import CLIBase
import argparse
from pathlib import Path


pattern = r'(?:cdef class )(.+)(?::)|((?:    def )[\s\S]+?(?::))(?:(\s*"""[\s\S]*?""")|\s*\n)|(    @[\s\S]*?\n)'
"""
Regular expression matching class, functions and decorators.
"""

func_pattern = r"(?:    def\s+?)(\w+?)(?:\()([\s\S]+?)(?:\))([\w\W]*?)(?::)"
param_pattern = r"(?:\s*?\w+\s+)?([\w\W]+?)$"


# TODO: Test generate_cython_pyi
# TODO: Make generate_cython_pyi more flexible. It's currently too limiting.

# List of requested features.
#  - Non class member functions.
#  - Class static variables.
#  - Automatic imports. Or at least a pragma to define these.
#  - Python 'class' definitions. Not only "cdef class".
#  - Empty class with pass in it.
#  - Cython properties.


def generate_cython_pyi(pyx_code: str) -> str:
    """
    Generates the class and function stubs for *.pyi files
    from Cython *.pyx code.

    :param pyx_code: String containing the code from the pyx file.
    :return: String containing the code for the pyi file.
    """

    stubs = StringIO()

    stubs.write("from typing import Sequence, Tuple, Dict, Optional, Union, Iterable\n"                
                "from os import PathLike\n"
                "import numpy as np\n\n")

    # Find and loop through all, class, function and decorators in the file.
    for match in re.finditer(pattern, pyx_code):
        if match.group(1) is not None:
            # Write cdef class as class.
            stubs.write(f"\nclass {match.group(1)}:\n")
        elif match.group(2) is not None:
            match_2 = match.group(2).replace("__cinit__", "__init__")

            # Find function name, docstring, arguments, their types and the return type.
            func_match = re.match(func_pattern, match_2)
            if func_match is None:
                continue
            func_name = func_match.group(1)
            func_params = func_match.group(2)
            func_params = [re.match(param_pattern, param).group(1) for param in func_params.split(",")]

            stubs.write(f"    def {func_name}({', '.join(func_params)}){func_match.group(3)}:{match.group(3) or ''}\n        ...\n\n")
        elif match.group(4) is not None:
            # Write decorator.
            stubs.write(match.group(4))

    return stubs.getvalue()


class Pyx2PyiCLI(CLIBase):
    @classmethod
    def configure_arg_parser(cls, sub_parsers):
        parser = sub_parsers.add_parser("x2i", help="Generates *.pyi files with stubs from Cython *.pyx file.")
        parser.add_argument("paths", nargs='+')
        parser.set_defaults(func=cls.main)

    @classmethod
    def main(cls, args: argparse.Namespace):
        # Loop through all paths.
        for path in args.paths:
            pyx_path = Path(path)
            pyi_path = pyx_path.with_suffix(".pyi")
            if pyx_path.is_file():
                with pyx_path.open("r") as f_pyx:
                    with pyi_path.open("w") as f_pyi:
                        # Read code in pyx.
                        # Call generate_cython_pyi.
                        # Write result to pyi file.
                        f_pyi.write(generate_cython_pyi(f_pyx.read()))
                print(f"{pyx_path} -> {pyi_path}")
            elif pyx_path.is_dir():
                for path in pyx_path.glob("*.pyx"):
                    pyi_path = path.with_suffix(".pyi")
                    with path.open("r") as f_pyx:
                        with pyi_path.open("w") as f_pyi:
                            f_pyi.write(generate_cython_pyi(f_pyx.read()))
                    print(f"{path} -> {pyi_path.name}")


