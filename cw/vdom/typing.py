from typing import Protocol, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .vdom import VDOM


class HTMLProtocol(Protocol):
    def __html__(self): ...


class ReprHTMLProtocol(Protocol):
    def _repr_html_(self): ...

