from io import StringIO
import numpy as np
import inspect
import yaml


def code_print(code: str, file=None, print_function=None):
    """
    Print function used print out code with line numbers.

    :param code: String containing the code to print.
    :param file:
    :param print_function: Print function to use. By default this
                           is the python print(...) function.
    :return:
    """
    str_io = StringIO()

    lines = code.splitlines()
    n_chars = int(np.floor(np.log10(len(lines))+1))

    for i, line in enumerate(lines):
        str_io.write(f"{i+1:{n_chars}}| {line}\n")

    (print_function or print)(str_io.getvalue(), file=file)


def debug_print(*args, sep=' ', file=None):
    """
    Print function that appends the printed text with the location of the print statement.
    
    :param args: 
    :param sep: 
    :param file: 
    :return: 
    """
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

    if verbose:
        return print_function or print
    else:
        def vprint(*args, **kwars):
            pass
        return vprint


def yaml_print(obj, default_flow_style=False, print_func=debug_print):
    """

    :param obj: Object to print
    :param default_flow_style: True to use PyYAML's default flow style. For more
                               information check PyYAML's documentation.
    :param print_func: Print function to use. By default its debug print.
    :return:
    """
    return print_func(yaml.dump(obj, default_flow_style=default_flow_style))


# Monkey patch debug print and yaml_print into the buildins so they are available everywhere.
__builtins__["debug_print"] = debug_print
__builtins__["yaml_print"] = yaml_print
