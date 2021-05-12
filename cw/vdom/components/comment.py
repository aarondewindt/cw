from markupsafe import Markup

def comment(comment):
    return Markup(f"<!-- {comment} -->")