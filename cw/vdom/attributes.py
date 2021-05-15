from typing import Any
from markupsafe import Markup, escape


class Attribute:
    def __init__(self):
        self._name = None
        self._value = None

    @classmethod
    def from_value(cls, name: str, value: Any):
        instance = cls()
        instance._name = name
        instance._value = value
        return instance

    @property
    def name(self) -> str:
        return self._name

    @property
    def value(self) -> Any:
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    @property
    def markup(self) -> Markup:
        if isinstance(self._value, bool):
            return escape(self._name).replace('_', '-') if self._value else ""
        else:
            return Markup(f'{escape(self._name).replace("_", "-")}="{escape(self._value)}"')


class Style(Attribute):
    def __init__(self, **kwargs):
        super().__init__()
        self._name = "style"
        self._value = kwargs

    @property
    def markup(self) -> Markup:
        inline_css = "; ".join(['{}: {}'.format(escape(k).replace("_", "-"), escape(v))
                                for k, v in self._value.items()])
        return Markup(f' style="{inline_css}"')

    def __getitem__(self, item):
        return self._value[item]

    def __setitem__(self, key, value):
        self._value[key] = value


def attr(name, value):
    return Attribute.from_value(name, value)


def element_style(**kwargs):
    return Style(**kwargs)


css = element_style


def data(name, value):
    return Attribute.from_value(f"data_{name}", value)
