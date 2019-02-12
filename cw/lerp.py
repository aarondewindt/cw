def lerp(x, x0, x1, y0, y1):
    """
    Linearly interpolate between points (x0, y0) and (x1, y1) to
    get the value of y at x.

    :param x:
    :param x0:
    :param x1:
    :param y0:
    :param y1:
    :return:
    """
    t = (x - x0) / (x1 - x0)
    return (1 - t) * y0 + t * y1
