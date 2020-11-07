from dataclasses import fields


class StatesBase:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_y_dot(self):
        raise NotImplementedError()

    def get_y(self):
        raise NotImplementedError()

    def set_t_y(self, t, y):
        raise NotImplementedError()

    @staticmethod
    def get_differentiation_y():
        return None

    @staticmethod
    def set_differentiation_y_dot(y_dot):
        pass

    def _repr_html_(self):
        html_lines = ["States:"]
        for field in fields(self):
            value = getattr(self, field.name)
            html_lines.append(f" - {field.name}: {value}")
        return "<br>".join(html_lines)
