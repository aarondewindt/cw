import argparse
import pathlib

from cw.flex_file import flex_dump, flex_load
from cw.cli_base import CLIBase
from cw.xsens import XSensParser
from cw.xsens.large_file_parser import XSensLargeFileParser


class XsensCLI(CLIBase):
    @classmethod
    def configure_arg_parser(cls, sub_parsers):
        parser = sub_parsers.add_parser("xsens", help="Decodes xsens logs")
        parser.add_argument("--input_file", "-i")
        parser.add_argument("--calibration_data", "-c")
        parser.add_argument("--output_file", "-o", action="append")
        parser.add_argument("--verbose", "-v", action="store_true")
        parser.add_argument("--large_file", "-l", action="store_true")
        parser.set_defaults(func=cls.main)

    @classmethod
    def main(cls, args: argparse.Namespace):
        # Load calibration data if provided.
        calibration_data = None
        if args.calibration_data is not None:
            calibration_data = flex_load(args.calibration_data)

        if args.input_file is not None:
            if args.large_file:
                output_path = pathlib.PurePath(args.output_file[0])
                output_format = "".join(output_path.suffixes)
                output_format = None if output_format == '' else output_format
                output_path = output_path.stem

                with open(args.input_file, "rb") as f:
                    xp = XSensLargeFileParser(f,
                                              output_path=output_path,
                                              output_format=output_format,
                                              calibration_data=calibration_data,
                                              verbose=args.verbose)
                    xp.start()

            else:
                with open(args.input_file, "rb") as f:
                    xp = XSensParser(f, verbose=args.verbose, calibration_data=calibration_data)
                    xp.start()
                    if args.verbose:
                        print("device_id:", xp.device_id)
                    tables = xp.get_tables()
                    for key, value in tables.items():
                        if isinstance(value, str):
                            continue
                        # print(key, type(value['time']), type(value['data'][0]))

                for output_path in map(pathlib.Path, args.output_file):
                    flex_dump(tables, output_path)

        else:
            raise ValueError("No input source given.")

