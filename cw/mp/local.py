from multiprocessing import Pool, set_start_method
import time

from collections import deque
import pickle
import pandas as pd
from tqdm import tqdm
import signal
import sys


from cw.mp.project import Project
from cw.mp.batch_configuration_base import BatchConfigurationBase


# I don't like having globals, but since all subprocesses need the same
# batch object we can save the constant pickling and unpickling of it
# by moving it to the global space before forking the new subprocesses.
# This is probably not threadsafe. So don't run run_project_locally(...)
# on multiple threads at the same time.
global_batch: BatchConfigurationBase = None
global_verbose = False


def run_project_locally(project: Project,
                        output_name: str,
                        n_cores: int,
                        dump_interval: int=5,
                        chunksize: int=1,
                        verbose=False,
                        ignore_directory_validity=False):
    global global_batch
    global global_verbose
    global_verbose = verbose

    # Try to set the multiprocessing start method to 'fork'. This is the
    # the fastest in our case, but it only works on unix systems
    try:
        set_start_method('fork')
        is_start_method_fork = True
    except:
        is_start_method_fork = False
        pass

    # Create the paths to the intermediate file path and output file.
    intermediate_file_path = project.path / f"{output_name}.int.pickle"
    output_file_path = project.path / f"{output_name}.pickle"

    # Get package batch object.
    batch = project.batch
    global_batch = batch

    # If an intermediate file exists, get the index of the last case that
    # was dumped into the intermediate result file.
    last_idx = None
    if intermediate_file_path.exists():
        if project.validate_directory() or ignore_directory_validity:
            with intermediate_file_path.open("rb") as f:
                # Loop to read all blocks in the pickle file stream and keep the last block
                last_result_block = None
                while True:
                    try:
                        last_result_block = pickle.load(f)
                    except EOFError:
                        break
                # If no blocks where found keep idx at None
                if last_result_block is not None:
                    last_idx = last_result_block[0]
        else:
            raise RuntimeError("Project code and/or data has been changed during an incomplete local run.")
    else:
        project.create_checksum_file(force=True)

    # Last time data was dumped to the intermediate file.
    last_dump_time = time.perf_counter()

    # Generator that yields the case index, case_input, dictionary and the batch object
    def cases():
        for i, case_input in enumerate(batch.create_cases()):
            if is_start_method_fork:
                yield i, tuple(case_input), None, verbose
            else:
                yield i, tuple(case_input), batch, verbose

    # Create iterator object from the generator
    cases_iter = iter(cases())

    # If a last index was found in the intermediate file.
    # skip all cases up to that one.
    if last_idx:
        for case in cases_iter:
            if case[0] >= last_idx:
                break

    result_block = deque()
    # Open processing pool and loop through the case inputs.
    with Pool(processes=n_cores) as pool:
        for case_result in tqdm(
                pool.imap(process_case, cases_iter, chunksize),
                initial=(last_idx or -1) + 1,
                total=batch.number_of_cases,
                disable=verbose):
            result_block.append(case_result)
            if (time.perf_counter() - last_dump_time) >= dump_interval:
                dump_block(intermediate_file_path, result_block)
                result_block = deque()
                last_dump_time = time.perf_counter()

    # Dump remaining results to the intermediate file.
    dump_block(intermediate_file_path, result_block)
    del result_block

    # Generator that yields the pandas dataframes from each block logged.
    def read_intermediate_file():
        with intermediate_file_path.open("rb") as f:
            while True:
                try:
                    # Load block
                    # Yield each row in the block
                    block = pickle.load(f)[1]
                    for case_idx, case_results in block:
                        yield case_results
                except EOFError:
                    break

    # Concatenate the individual data block in the intermediate file together.
    result = pd.DataFrame(read_intermediate_file(), columns=(*batch.input_parameters, *batch.output_parameters))

    # Dump the data into the output file.
    with output_file_path.open("wb") as f:
        pickle.dump(result, f)

    # Delete intermediate file.
    intermediate_file_path.unlink()

    return result


def dump_block(path, block):
    if len(block):
        last_idx = block[-1][0]
        with path.open("ab") as f:
            pickle.dump((last_idx, block), f)


def process_case(case):
    global global_batch

    i, case_input, batch, verbose = case

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if not verbose:
        sys.stdout = write_null

    batch = batch or global_batch

    # Run case
    case_output = batch.run_case(batch.InputTuple(*case_input))
    # Merge input and output tuples into a single tuple.
    case_result = (*case_input, *case_output)
    return i, case_result


class WriteNull:
    def write(self, *args, **kwargs):
        pass

    flush = write


write_null = WriteNull()
