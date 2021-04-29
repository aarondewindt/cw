from typing import Dict, Any
from functools import lru_cache
from pathlib import Path
import random
import string

from ..html import style_, div, section, input_, article, label

# From: https://codepen.io/CameronSchuyler/pen/OpOpWo
vertical_tabs_css = """
"""

# @lru_cache(None)
def load_css():
    with (Path(__file__).parent / "accordion.css").open("r") as f:
        return style_(f.read())

def random_string():
    """

    :return: string with 10 random lower case characters
    """
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))

def accordion(content: Dict[Any, Any], multiple=True):
    """
    Accordion component.

    :param content: Dictionary whose keys are the section names and the value the section content.
    :param multiple: True to allow multiple sections to be open at the same time.
    :return: Accordion component.
    """
    name = random_string()
    input_type = "checkbox" if multiple else "radio"
    def sections():
        for i, (key, value) in enumerate(content.items()):
            option_id = random_string()
            yield section(
                input_(name=name, id=option_id, type=input_type, Class="sections", checked="true")
                    if i == 0 else input_(name=name, id=option_id, type=input_type, Class="sections"),
                label(For=option_id, c=key),
                article(value)
            )

    return div(
        load_css(),
        div(Class="cw_vdom_accordion", c=list(sections()))
    )



