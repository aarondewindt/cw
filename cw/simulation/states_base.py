from dataclasses import fields

from IPython.display import display, JSON


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

    def _ipython_display_(self):
        fields_dict = {field.name: str(getattr(self, field.name)) for field in fields(self)}
        display(JSON(fields_dict))
