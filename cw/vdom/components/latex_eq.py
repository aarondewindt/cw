from markupsafe import Markup


def latex_eq(equation, displayed_mode=False):
    """
    Wrap the equation around `$` or `$$` whether it needs to be in displayed mode or not.

    :param equation:
    :param displayed_mode:
    :return:
    """
    if displayed_mode:
        return Markup(f"$${equation}$$")
    else:
        return Markup(f"${equation}$")