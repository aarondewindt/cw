from io import StringIO
from typing import Dict, Sequence, Optional, Union

from markupsafe import Markup, escape
import html2text

from .typing import HTMLProtocol, ReprHTMLProtocol
from .attributes import Attribute, Style


class VDOM:
    def __init__(self,
                 tag_name: str,
                 attributes: Optional[Dict[str, Attribute]]=None,
                 children: Optional[Sequence[Union['VDOM', HTMLProtocol, ReprHTMLProtocol, str]]]=None):
        self.tag_name = tag_name
        self.attributes = attributes or {}
        self.children = children or []

    def to_html(self):
        with StringIO() as out:

            out.write(f"<{escape(self.tag_name)}")

            # Write attributes
            for _, attribute in self.attributes.items():
                out.write(" ")
                out.write(attribute.markup)

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


def create_component(tag_name: str, allow_children=True):
    def _component(*children: Sequence[Union['VDOM', HTMLProtocol, ReprHTMLProtocol, str, Attribute]],
                   c: Optional[Sequence[Union['VDOM', HTMLProtocol, ReprHTMLProtocol, str, Attribute]]]=None,
                   **attributes: Dict[str, Union[str, bool, HTMLProtocol]]):
        # Combine the list of children from the args and `c` arguments.
        children = list(children)
        if c:
            children.extend(c)

        # Create attribute objects from the kwargs.
        attribute_objects = {
            key: Style(**value) if key == "style" else Attribute.from_value(key, value)
            for key, value in attributes.items()}

        # Move all attributes objects in the children to attribute objects dictionary
        for child in tuple(children):
            if isinstance(child, Attribute):
                attribute_objects[child.name] = child
                children.remove(child)

        if not allow_children and children:
            raise ValueError(f"<{tag_name}/> cannot have children")

        return VDOM(tag_name, attribute_objects, children)

    return _component


def h(tag_name, *children, **kwargs):
    el = create_component(tag_name)
    return el(*children, **kwargs)
