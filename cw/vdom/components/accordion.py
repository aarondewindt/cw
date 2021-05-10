from typing import Dict, Any
from functools import lru_cache
from pathlib import Path
import random
import string

from ..html import style, div, section, input_, article, label

# From: https://codepen.io/CameronSchuyler/pen/OpOpWo


# @lru_cache(None)
def load_css():
    with (Path(__file__).parent / "accordion.css").open("r") as f:
        return style(f.read())


def random_string():
    """

    :return: string with 10 random lower case characters
    """
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))


def accordion(content: Dict[Any, Any], multiple=True):
    """
    Accordion component.
    :param content: Dictionary whose keys are the section titles and the value the section content. The keys
                    may be a two element tuples whose first element is the section title and the second
                    element a highlight level. The level may be `brand`, `accent`, `warn`, `error`,
                    `success` or `info`.
    :param multiple: True to allow multiple sections to be open at the same time.
    :return: Accordion component.
    """
    accordion_name = random_string()
    input_type = "checkbox" if multiple else "radio"
    def sections():
        for i, (key, value) in enumerate(content.items()):
            label_class = ""
            if isinstance(key, (list, tuple)):
                assert len(key) == 2, "The key may either be a single object or tuple/list with two elements"
                key, level = key
                label_class = f"cw_vdom_{level}"

            option_id = random_string()
            yield section(
                input_(name=accordion_name, id=option_id, type=input_type, Class="sections"),
                label(For=option_id, Class=label_class, c=[key]),
                article(value)
            )

    return div(
        load_css(),
        div(Class="cw_vdom_accordion", c=list(sections()))
    )



