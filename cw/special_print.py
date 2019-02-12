from io import StringIO
import numpy as np
import inspect


def print_code(code: str, file=None):
    """
    Print function used print out code with line numbers.

    :param code: String containing the code to print.
    :param file:
    :return:
    """
    str_io = StringIO()

    lines = code.splitlines()
    n_chars = int(np.ceil(np.log10(100))+2)

    for i, line in enumerate(lines):
        str_io.write(f"{i+1: {n_chars}}| {line}\n")

    print(str_io.getvalue(), file=file)


def debug_print(*args, sep=' ', file=None):
    """
    Print function that appends the printed text with the location of the print statement.
    
    :param args: 
    :param sep: 
    :param file: 
    :return: 
    """""
    file_name = inspect.stack()[1][1]
    line_no = inspect.stack()[1][2]
    function_name = inspect.stack()[1][3]
    msg = sep.join(map(str, args))
    print(f'{msg}: File "{file_name}", line {line_no}, in {function_name}', file=file)


def verbose_print(verbose, print_function=None):
    """
    Factory function that return a working print function if verbose is True.
    Otherwise it returns a dummy function that won't print anything.

    :param verbose: True to return print_function
    :param print_function: Optional print function to use. Uses 'print' by default.
    :return:
    """
    print_function = print_function or print
    if verbose:
        return print_function
    else:
        def vprint(*args, **kwars):
            pass
        return vprint()


# Monkey patch debug print into the buildins to it's available everywhere.
__builtins__["debug_print"] = debug_print

