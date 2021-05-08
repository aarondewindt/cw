import random
import string
from collections import deque

from typing import Dict, Any
import numpy as np
from functools import lru_cache
from pathlib import Path
from markupsafe import escape


from ..html import style_, div, section, input_, article, label, pre, span, br
from .safe import safe
from .tabulate import tabulate
from ..attributes import element_style


@lru_cache(None)
def load_css():
    with (Path(__file__).parent / "hyr.css").open("r") as f:
        return style_(safe(f.read()))


def random_string():
    """

    :return: string with 10 random lower case characters
    """
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))


def hyr(content, show_type=True):
    _, tp, element = hyr_process(content, show_type)
    return div(load_css(), tp, element, Class="cw_vdom_hyr")


def fullname(o):
    klass = o.__class__
    module = klass.__module__
    if module == 'builtins':
        return klass.__qualname__
    return module + '.' + klass.__qualname__


def hyr_process(content, show_type):
    if show_type:
        type_element = span(f"{fullname(content)}:", Class="cw_vdom_hyr_type")
    else:
        type_element = ""

    if isinstance(content, dict):
        def process_element(key: str):
            option_id = random_string()
            collapsable, tp, element_content = hyr_process(content[key], show_type)
            return div(
                input_(type="checkbox", id=option_id) if collapsable else "",
                label(f"{key}: ", For=option_id), tp,
                div(element_content),
                Class="cw_vdom_hyr_item")

        return True, type_element, div(*[process_element(key) for key in content])

    elif isinstance(content, str):
        multiline = "\n" in content
        escaped_content = safe(str(escape(content)).replace("\n", "<br/>"))
        if multiline:
            return True, type_element, div(escaped_content)
        else:
            return False, type_element, span(escaped_content)

    # Ordered iterables
    elif isinstance(content, (list, tuple, deque)):
        def process_element(idx):
            option_id = random_string()
            collapsable, tp, element_content = hyr_process(content[idx], show_type)
            return div(
                input_(type="checkbox", id=option_id) if collapsable else "",
                label(f"[{idx}] ", For=option_id), tp,
                div(element_content),
                Class="cw_vdom_hyr_item")

        return True, type_element, div(*[process_element(idx) for idx in range(len(content))])

    # Unordered iterables
    elif isinstance(content, set):
        def process_element(element):
            option_id = random_string()
            collapsable, tp, element_content = hyr_process(element, show_type)
            return div(
                input_(type="checkbox", id=option_id) if collapsable else "",
                label(f"[] ", For=option_id), tp,
                div(element_content),
                Class="cw_vdom_hyr_item")

        return True, type_element, div(*[process_element(element) for element in content])

    elif isinstance(content, np.ndarray):
        if content.ndim == 1 or (content.ndim == 2 and content.shape[0] == 1):
            return False, type_element, escape(content)
        else:
            return True, type_element, div(safe(str(escape(content)).replace("\n", "<br/>")))

    elif hasattr(content, "_repr_html_") or hasattr(content, "__html__"):
        return True, type_element, div(content)

    else:
        escaped_content = safe(str(escape(content)).replace("\n", "<br/>"))
        if "<br/>" in escaped_content:
            return True, type_element, div(escaped_content)
        else:
            return False, type_element, span(escaped_content)


