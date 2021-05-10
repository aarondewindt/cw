from typing import Any
from markupsafe import Markup


def safe(content: Any):
    """
    Mark text as safe. It will not be escaped when rendering the HTML code.
    If the object has a `_repr_html_` or `__html__` function it will returned
    without making changes.

    :param content: Content to make as safe.
    :return: A `markupsafe.Markup` string.
    """
    if hasattr(content, "_repr_html_"):
        return content
    elif hasattr(content, "__html__"):
        return content
    else:
        return Markup(content)
