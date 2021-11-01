"""
Virtual DOM package heavily based on https://github.com/nteract/vdom
"""
from .vdom import VDOM, create_component, h
from .html import *
from .components.safe import safe
from .components.latex_eq import latex_eq
from .components.tabulate import tabulate
from .components.accordion import accordion
from .components.comment import comment
from .components.hyr import hyr
from .attributes import attr, element_style, data, css

nl = safe("\n")
