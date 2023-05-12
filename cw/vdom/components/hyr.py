import random
import string
from collections.abc import Iterable, Sized, Sequence, Collection

from typing import Dict, Any
import numpy as np
import xarray as xr
from functools import lru_cache
from pathlib import Path
from markupsafe import escape


from ..html import style, div, input_, label, span
from .safe import safe
from ..vdom import VDOM


@lru_cache(None)
def load_css():
    with (Path(__file__).parent / "hyr.css").open("r") as f:
        return style(safe(f.read()))


def random_string():
    """

    :return: string with 10 random lower case characters
    """
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))


def hyr(content, show_type=True, title="", root_type=None, top_n_open=1):
    collapsed = not bool(top_n_open)
    top_n_open = max(0, top_n_open - 1)
    _, tp, element = hyr_process(content, show_type, top_n_open)

    if root_type is None:
        root_type = tp
    else:
        root_type = span(f"{type_fullname(root_type)}:", Class="cw_vdom_hyr_type")

    option_id = random_string()
    return div(load_css(),
               input_(type="checkbox", id=option_id, checked=not collapsed),
               label(title, For=option_id), root_type if show_type else "",
               div(element),
               Class="cw_vdom_hyr cw_vdom_hyr_item")


def type_fullname(klass):
    module = klass.__module__
    if module == 'builtins':
        return klass.__qualname__
    return module + '.' + klass.__qualname__


def hyr_process(content, show_type, top_n_open):
    collapsed = not bool(top_n_open)
    top_n_open = max(0, top_n_open - 1)

    if show_type:
        if isinstance(content, Iterable):
            if isinstance(content, Sized):
                try:
                    type_element = span(f"{type_fullname(content.__class__)}[{len(content)}]:",
                                        Class="cw_vdom_hyr_type")
                except TypeError:
                    type_element = span(f"{type_fullname(content.__class__)}[]:",
                                        Class="cw_vdom_hyr_type")
            else:
                type_element = span(f"{type_fullname(content.__class__)}[]:", Class="cw_vdom_hyr_type")
        else:
            type_element = span(f"{type_fullname(content.__class__)}:", Class="cw_vdom_hyr_type")
    else:
        type_element = ""

    if isinstance(content, dict):
        def process_element(key: str):
            option_id = random_string()
            collapsable, tp, element_content = hyr_process(content[key], show_type, top_n_open)
            return div(
                input_(type="checkbox", id=option_id, checked=not collapsed) if collapsable else "",
                label(key, For=option_id), tp,
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

    elif isinstance(content, np.ndarray):
        if content.ndim == 1 or (content.ndim == 2 and content.shape[0] == 1):
            return False, type_element, escape(content)
        else:
            return True, type_element, div(safe(str(escape(content)).replace("\n", "<br/>")))

    elif hasattr(content, "_repr_html_") or hasattr(content, "__html__"):
        return True, type_element, div(content)

    # Iterables
    elif isinstance(content, Iterable):
        # Ordered iterables
        if isinstance(content, Sequence):
            def process_element(idx):
                option_id = random_string()
                collapsable, tp, element_content = hyr_process(content[idx], show_type, top_n_open)
                return div(
                    input_(type="checkbox", id=option_id, checked=not collapsed) if collapsable else "",
                    label(f"[{idx}]", For=option_id), tp,
                    div(element_content),
                    Class="cw_vdom_hyr_item")

            return True, type_element, div(*[process_element(idx) for idx in range(len(content))])

        # Unordered iterables
        elif isinstance(content, Collection):
            def process_element(element):
                option_id = random_string()
                collapsable, tp, element_content = hyr_process(element, show_type, top_n_open)
                return div(
                    input_(type="checkbox", id=option_id, checked=not collapsed) if collapsable else "",
                    label(f"[]", For=option_id), tp,
                    div(element_content),
                    Class="cw_vdom_hyr_item")

            return True, type_element, div(*[process_element(element) for element in content])
        else:
            return False, type_element, div("... Unsized iterable, may contain infinite values.")

    else:
        escaped_content = safe(str(escape(content)).replace("\n", "<br/>"))
        if "<br/>" in escaped_content:
            return True, type_element, div(escaped_content)
        else:
            return False, type_element, span(escaped_content)


