from io import StringIO
import numpy as np


def print_code(code: str):
    str_io = StringIO()

    lines = code.splitlines()
    n_chars = int(np.ceil(np.log10(100))+2)

    for i, line in enumerate(lines):
        str_io.write(f"{i+1: {n_chars}}| {line}\n")

    print(str_io.getvalue())
