from typing import Union, Optional

from markupsafe import Markup

from .typing import HTMLProtocol, ReprHTMLProtocol
from .html import table, tr, th, td, tbody, thead, tfoot, caption as caption_component
import numpy as np


def safe(content: Union[HTMLProtocol, ReprHTMLProtocol, str]):
    """
    Mark text as safe. It will not be escaped when rendering the HTML code.
    If the object has a `_repr_html_` or `__html__` function it will returned
    without making changes.

    :param content: Content to make as safe.
    :return: A `markupsafe.Markup` string.
    """
    if hasattr(content, "_repr_html_"):
        return content
    elif hasattr(content, "__html__"):
        return content
    else:
        return Markup(content)


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


def tabulate(data, header=None, row_header=None, caption=None):
    """
    Create table.

    :param data: Data in the table body.
    :param header: Column headers
    :param row_header: Row headers
    :param caption: Caption to add on top of the table
    :return: `cw.vdom.table` component with the new table.
    """
    # Make sure the data is 2d.
    data = np.atleast_2d(data)

    has_header = header is not None
    has_row_header = row_header is not None

    if has_row_header:
        # Make sure the data is 2d.
        # We can't use numpy's `atleast_2d` function since that will
        # make it a row vector and we need it to be a column vector.
        row_header = np.asarray(row_header)
        if row_header.ndim == 1:
            row_header = np.expand_dims(row_header, -1)

        # Check size
        assert data.shape[0] == row_header.shape[0], "Data and row headers must have the same number of rows."
    else:
        # If we were given no row headers make a new matrix with zero columns
        row_header = np.empty((data.shape[0], 0), dtype=np.unicode)

    if has_header:
        header = np.atleast_2d(header)
        assert data.shape[1] == header.shape[1], "Data and headers must have the same number of columns."

        head = thead(
            # Loop through each row in the header
            *(tr(
                # Add one empty `th` for each column in the row_headers
                *(th() for _ in range(row_header.shape[1])),

                # Add one `th` for each column in the header
                *(th(head, scope="col") for head in row)

              ) for row in header))
    else:
        head = ""

    return table(
        caption_component(caption) if caption else "",
        head,
        tbody(
            # Loop through each row in the data/row_header
            *(tr(
                # Add one `th` for each column in the row_headers
                *(th(h, scope="row") for h in row_h),

                # Add one `td` for each column in the data
                *(td(d) for d in row_data)

            ) for row_h, row_data in zip(row_header, data))
        ),
    )
