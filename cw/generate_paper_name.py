import re

from cw.cli_base import CLIBase


def generate_paper_name(first_author, title):
    title = "_".join(title.strip().lower().split())
    first_author = re.sub(r"(?:\s*)(\w)(?:[\w\W]*)\s(\S+)", r"\1_\2", first_author.strip().lower())

    print(f"{first_author}_{title}")
