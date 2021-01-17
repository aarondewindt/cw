from typing import Union

from markupsafe import Markup

from .typing import HTMLProtocol, ReprHTMLProtocol


def safe(content: Union[HTMLProtocol, ReprHTMLProtocol, str]):
    if hasattr(content, "_repr_html_"):
        return Markup(content._repr_html_())
    else:
        return Markup(content)


def latex_eq(equation, displayed_mode=False):
    if displayed_mode:
        return Markup(f"$${equation}$$")
    else:
        return Markup(f"${equation}$")
