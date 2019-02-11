import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="cw",
        description="CW command line tools.",
    )
    subparsers = parser.add_subparsers()

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
