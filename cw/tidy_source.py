def tidy_source(code):
    """
    Strips and fixes the indentation of python code.
    eg.

    |    def foo():
    |        bar()

    or

    |def foo():
    |        bar()

    becomes

    |def foo():
    |    bar()
    """
    raw_lines = code.splitlines()
    lines = []

    # Remove all empty lines
    for line in raw_lines:
        if line.strip() != "":
            lines.append(line)

    if len(lines) > 1:
        if lines[0].rstrip()[-1] == ":":
            indents = 4  # If you don't use 4 spaces, you will break stuff.
        else:
            indents = 0
        lines[0] = lines[0].lstrip()
        remove = len(lines[1])-len(lines[1].lstrip())-indents
        code = lines[0]
        for idx in range(1, len(lines)):
                lines[idx] = lines[idx][remove:]
                code = "\n".join((code, lines[idx]))
    else:
        code = code.strip()
    return code