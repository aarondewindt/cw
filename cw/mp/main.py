from cw.cli_base import CLIBase
import argparse
from pathlib import Path
import os
from textwrap import indent
import multiprocessing

from cw.mp.project import Project
from cw.mp.local import run_project_locally


class MPCLI(CLIBase):
    main_parser = None
    @classmethod
    def configure_arg_parser(cls, sub_parsers):
        cls.main_parser = sub_parsers.add_parser("mp", help="Multiprocessing tools.")
        cls.main_parser.set_defaults(func=cls.mp)
        mp_subparsers = cls.main_parser.add_subparsers()

        init_parser = mp_subparsers.add_parser("init", help="Initializes a new cw.mp project.")
        init_parser.add_argument("path", help="Path of directory to initialize. Default is the current directory.",
                                 nargs='?', default=os.getcwd())
        init_parser.add_argument("--name", "-n", help="Batch name", default=None)
        init_parser.set_defaults(func=cls.init)

        info_parser = mp_subparsers.add_parser("info", help="Print the current cw.mp configuration.")
        info_parser.add_argument("path", help="Path of cw.mp project directory.",
                                 nargs='?', default=os.getcwd())
        info_parser.set_defaults(func=cls.info)

        run_parser = mp_subparsers.add_parser("run", help="Run batch.")
        run_parser.add_argument("path", help="Path of cw.mp project directory.", nargs='?', default=os.getcwd())
        # run_parser.add_argument("--number", "-n", help="Number of cases to run. By default all cases will run.",
        #                         default=None)
        run_parser.add_argument("--output_name", "-o", help="Name of the output file. (default is the batch name)")
        run_parser.add_argument("--n_cores", "-c", type=int, default=None,
                                help="Number of processes to start.")
        run_parser.add_argument("--dump_interval", "-i", type=int, default=5,
                                help="Seconds between dumps to the intermediate results file. (default=5)")
        run_parser.add_argument("--chunksize", "-s", type=int, default=1,
                                help="Number of cases to give each processes per time. (default=1)")
        run_parser.set_defaults(func=cls.run)

        clear_parser = mp_subparsers.add_parser("clear",
                                               help="Clears the intermediate data for an incomplete local run.")
        clear_parser.add_argument("path", help="Path of cw.mp project directory.",
                                 nargs='?', default=os.getcwd())
        clear_parser.add_argument("--output_name", "-o", help="Name of the output file. (default is the batch name)")
        clear_parser.set_defaults(func=cls.clear)

    @classmethod
    def mp(cls, args: argparse.Namespace):
        if cls.main_parser:
            cls.main_parser.print_help()

    @classmethod
    def init(cls, args: argparse.Namespace):
        try:
            Project.initialize(args.path, args.name)
        except ValueError as e:
            print("Error:", e)
        else:
            print(f"New cw.mp project initialized at '{args.path}'.")

    @classmethod
    def clear(cls, args: argparse.Namespace):
        try:
            project = Project(args.path)
        except ValueError as e:
            print("Error:", e)
        else:
            output_name = args.output_name or project.batch.name
            intermediate_file_path = project.path / f"{output_name}.int.pickle"
            if intermediate_file_path.exists():
                intermediate_file_path.unlink()
                print(f"Deleted '{intermediate_file_path}'")
            else:
                print(f"Already cleared '{intermediate_file_path}'")

    @classmethod
    def info(cls, args: argparse.Namespace):
        try:
            project = Project(args.path)
        except ValueError as e:
            print("Error:", e)
        else:
            input_param_str = indent('\n'.join(project.batch.input_parameters), " - ")
            output_param_str = indent('\n'.join(project.batch.output_parameters), " - ")
            print(f"path: {project.path}\n"
                  f"project name: {project.batch.name}\n"
                  f"batch name: {project.batch.name}\n"
                  f"number of cases: {project.batch.number_of_cases}\n"
                  f"batch description:\n{indent(project.batch.description_trimmed, '   ')}\n"
                  f"inputs: \n{input_param_str}\n"
                  f"outputs: \n{output_param_str}"
                  )

    @classmethod
    def run(cls, args: argparse.Namespace):
        try:
            project = Project(args.path)
        except ValueError as e:
            print("Error:", e)
        else:
            n_cores = int(args.n_cores or 0)
            max_cores = multiprocessing.cpu_count()
            n_cores = max_cores if (n_cores >= max_cores) or (n_cores < 0) else n_cores
            while (n_cores <= 0) or (n_cores > max_cores):
                try:
                    n_cores = int(input(f"Number of cores [1-{max_cores}]:\n"))
                except:
                    pass

            output_name = args.output_name or project.batch.name
            run_project_locally(project, output_name, n_cores, args.dump_interval, args.chunksize)
