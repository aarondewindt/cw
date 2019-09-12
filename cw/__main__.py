import argparse
from cw.generate_cython_pyi import Pyx2PyiCLI
from cw.mp.main import MPCLI


def main():
    parser = argparse.ArgumentParser(
        prog="cw",
        description="CW command line tools.",
    )
    subparsers = parser.add_subparsers()

    Pyx2PyiCLI.configure_arg_parser(subparsers)
    MPCLI.configure_arg_parser(subparsers)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
