from markupsafe import Markup

import sympy as sp

try:
    from sympy.core._print_helpers import Printable as PatchClass
except ImportError:
    try:
        from sympy.core import Basic as PatchClass
    except ImportError:
        PatchClass = None


if PatchClass is not None:
    # Make sympy objects escapable by markupsafe
    def sympy__html__(self):
        return Markup(f"${sp.latex(self)}$")

    setattr(PatchClass, "__html__", sympy__html__)
