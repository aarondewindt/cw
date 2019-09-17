

class StatesBase:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_y_dot(self):
        raise NotImplemented()

    def get_y(self):
        raise NotImplemented()

    def set_t_y(self, t, y):
        raise NotImplemented()
