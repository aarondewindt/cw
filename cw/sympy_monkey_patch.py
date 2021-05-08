from markupsafe import Markup

import sympy as sp
from sympy.core._print_helpers import Printable


# Make sympy objects escapable by markupsafe
def sympy__html__(self):
    return Markup(f"${sp.latex(self)}$")

setattr(Printable, "__html__", sympy__html__)
