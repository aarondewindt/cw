from io import StringIO
from typing import Dict, Sequence, Optional, Union

from markupsafe import Markup, escape
import html2text

from .typing import HTMLProtocol, ReprHTMLProtocol


class VDOM:
    def __init__(self, tag_name: str,
                       attributes: Optional[Dict[str, Union[str, bool, HTMLProtocol]]]=None,
                       style: Optional[Dict[str, str]]=None,
                       children: Optional[Sequence[Union['VDOM', HTMLProtocol, ReprHTMLProtocol, str]]]=None):
        self.tag_name = tag_name
        self.attributes = attributes or {}
        self.style = style or {}
        self.children = children or []

    def to_html(self):
        with StringIO() as out:
            out.write(f"<{escape(self.tag_name)}")

            if self.style:
                out.write(f' style="{to_inline_css(self.style)}"')

            for key, value in self.attributes.items():
                if isinstance(value, bool):
                    if value:
                        out.write(f" {escape(key).replace('_', '-')}")
                else:
                    out.write(f' {escape(key).replace("_", "-")}="{escape(value)}"')

            out.write('>')

            for child in self.children:
                if isinstance(child, VDOM):
                    out.write(child._repr_html_())
                elif hasattr(child, "_repr_html_"):
                    out.write(Markup(child._repr_html_()))
                else:
                    out.write(escape(child))

            out.write(f'</{escape(self.tag_name)}>')

            return out.getvalue()

    __html__ = to_html
    _repr_html_ = to_html

    def __repr__(self):
        return html2text.html2text(self.to_html())


def to_inline_css(style: Dict[str, str]):
    """
    Return inline CSS from CSS key / values
    """
    return "; ".join(['{}: {}'.format(escape(k).replace("_", "-"), escape(v)) for k, v in style.items()])


def create_component(tag_name: str, allow_children=True):
    def _component(*children: Sequence[Union['VDOM', HTMLProtocol, ReprHTMLProtocol, str]],
                   style: Optional[Dict[str, str]]=None,
                   c: Optional[Sequence[Union['VDOM', HTMLProtocol, ReprHTMLProtocol, str]]]=None,
                   **attributes):

        children = list(children)
        if c:
            children.extend(c)

        if not allow_children and children:
            raise ValueError(f"<{tag_name}/> cannot have children")

        return VDOM(tag_name, attributes, style, children)

    return _component


def h(tag_name, *children, **kwargs):
    el = create_component(tag_name)
    return el(*children, **kwargs)
