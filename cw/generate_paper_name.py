import re


replace_table = (
    ("-", "_"),
    ("–", "_")
)


def generate_paper_name(first_author: str, title: str):
    for replace_pair in replace_table:
        first_author = first_author.replace(*replace_pair)
        title = title.replace(*replace_pair)

    title = "_".join(title.strip().lower().split())
    first_author = re.sub(r"(?:\s*)(\w)(?:[\w\W]*)\s(\S+)", r"\1_\2", first_author.strip().lower())

    print(f"{first_author}_{title}")
