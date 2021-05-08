from .html import html, head, body, title as title_element


def html_page(title, head_contents=None, *args, **kwargs):
    """
    Full html page component

    :param title: Page title
    :param head_contents: Sequence of html elements to add to the head.
    :param args: Passed to the body.
    :param kwargs: Passed to the body.
    :return:
    """
    page = html(
        head(
            title_element(title),
            *head_contents,
        ),
        body(*args, **kwargs)
    )
    return page
